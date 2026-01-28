import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # pad if mismatch (safety)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class RLocUNetGaussian(nn.Module):
    """
    U-Net backbone + global pooling + FC heads to output (mu, log_var).
    Input: (B, C, 64, 64) with C=3 (Re, Im, beamwidth_map)
    Output: mu (B,), log_var (B,)
    """
    def __init__(self, in_ch=3, base=64, depth=4, dropout=0.0):
        super().__init__()
        # For simplicity, fixed 4-depth U-Net like 64->128->256->512
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)

        self.bottleneck = DoubleConv(base*8, base*16)

        self.up3 = Up(base*16, base*8, base*8)  # x5(1024) + skip x4(512) -> out 512
        self.up2 = Up(base*8,  base*4, base*4)  # 512 + 256 -> 256
        self.up1 = Up(base*4,  base*2, base*2)  # 256 + 128 -> 128
        self.up0 = Up(base*2,  base,   base)    # 128 + 64  -> 64

        # global pooling to 1x1 + FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        hid = base
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hid, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hid, 1)
        self.logvar_head = nn.Linear(hid, 1)

    def forward(self, x):
        x1 = self.inc(x)       # base
        x2 = self.down1(x1)    # base*2
        x3 = self.down2(x2)    # base*4
        x4 = self.down3(x3)    # base*8

        x5 = self.bottleneck(x4)  # base*16

        x = self.up3(x5, x4)
        x = self.up2(x,  x3)
        x = self.up1(x,  x2)
        x = self.up0(x,  x1)

        feat = self.pool(x)  # (B, base, 1, 1)
        feat = self.fc(feat)

        mu = self.mu_head(feat).squeeze(-1)
        logvar = self.logvar_head(feat).squeeze(-1)

        # clamp logvar to avoid numerical issues
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

def gaussian_nll_loss(y, mu, logvar):
    """
    Heteroscedastic Gaussian NLL (equivalent to KL form up to constant):
      0.5 * (logvar + (y-mu)^2 / exp(logvar))
    """
    var = torch.exp(logvar)
    return 0.5 * (logvar + (y - mu) ** 2 / (var + 1e-12))

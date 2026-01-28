import os
import re
import glob
import numpy as np
# from scipy.io import loadmat
import mat73
import torch
from torch.utils.data import Dataset

# -----------------------------
# Filename parsing (README)
#   "RoomName_APName_VolunteerID_InterferenceState"
#   e.g., "Lounge_sRE4_user1_w.mat"
#         "Lab_sRE5_user5_wo.mat"
# -----------------------------
FNAME_RE = re.compile(
    r"^(?P<room>Con|Lab|Office|Lounge)_(?P<ap>sRE\d+)_user(?P<user>\d+)_(?P<intf>w|wo)\.mat$"
)

def parse_rloc_filename(path):
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    d["user"] = int(d["user"])
    # normalize room naming if you want
    return d  # dict: room, ap, user, intf

def _safe_squeeze(x):
    return np.asarray(x).squeeze()

def build_aoa_tof_map_from_csi(
    csi_1x90,
    n_ant=3,
    n_sc=30,
    gA=64,
    gD=64,
    normalize=True,
):
    """
    Same as before: build a complex AoA-ToF map X (gA x gD) from CSI (length 90).
    AoA axis: FFT over antenna dimension (zero padded to gA)
    ToF axis: IFFT over subcarrier dimension (zero padded to gD)
    """
    csi = np.asarray(csi_1x90)

    # common cases from MATLAB:
    # (90,), (1,90), (90,1)
    csi = csi.reshape(-1)
    if csi.shape[0] != n_ant * n_sc:
        raise ValueError(f"Unexpected CSI length: {csi.shape[0]} (expect {n_ant*n_sc})")

    csi = csi.reshape(n_ant, n_sc)
    if not np.iscomplexobj(csi):
        csi = csi.astype(np.complex64)

    aoa_spec = np.fft.fft(csi, n=gA, axis=0)
    aoa_spec = np.fft.fftshift(aoa_spec, axes=0)

    tof_spec = np.fft.ifft(aoa_spec, n=gD, axis=1)
    tof_spec = np.fft.fftshift(tof_spec, axes=1)

    X = tof_spec.astype(np.complex64)

    # beamwidth estimate from AoA marginal power
    power_aoa = np.sum(np.abs(X) ** 2, axis=1)  # (gA,)
    peak = int(np.argmax(power_aoa))
    half = power_aoa[peak] * 0.5
    left = peak
    while left - 1 >= 0 and power_aoa[left - 1] >= half:
        left -= 1
    right = peak
    while right + 1 < gA and power_aoa[right + 1] >= half:
        right += 1
    beamwidth = float(right - left + 1)

    if normalize:
        mag_rms = np.sqrt(np.mean(np.abs(X) ** 2) + 1e-12)
        X = X / mag_rms

    return X, beamwidth

class HWILDDataset(Dataset):
    """
    H-WILD dataset loader following README:

    File naming:
      Room_AP_userX_w/wo.mat

    Variables inside each .mat (README):
      - estimations_aoa   : 2D-FFT estimated angles (optional for training)
      - features_csi      : 1x90 (3 ants x 30 subcarriers) per packet
      - features_rssi     : 1x3
      - features_agc      : ...
      - labels_aoa        : AoA label per packet
      - uwb_coordinate_x/y: position GT (optional)

    This loader supports:
      - filtering by room/ap/user/intf
      - different split strategies:
          * "random" (sample-level)
          * "leave_one_user_out"
          * "cross_room"
          * "cross_ap"
    """
    def __init__(
        self,
        data_root,
        split="train",
        # ---- filters ----
        rooms=None,          # e.g. ["Lounge","Office"] or ["Con","Lab"]
        aps=None,            # e.g. ["sRE4","sRE5"]
        users=None,          # e.g. [1,2,3]
        interference=None,   # "w" or "wo" or None for both
        # ---- split ----
        split_strategy="random",
        train_ratio=0.8,         # used for random
        held_out_user=None,      # used for leave_one_user_out
        held_out_room=None,      # used for cross_room
        held_out_ap=None,        # used for cross_ap
        seed=42,
        # ---- representation ----
        gA=64,
        gD=64,
        angle_unit="deg",
        include_rssi_agc=False,  # optional: add rssi/agc as extra channels
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.split_strategy = split_strategy
        self.train_ratio = train_ratio
        self.seed = seed

        self.gA = gA
        self.gD = gD
        self.angle_unit = angle_unit
        self.include_rssi_agc = include_rssi_agc

        # normalize filter values
        self.rooms = set(rooms) if rooms else None
        self.aps = set(aps) if aps else None
        self.users = set(users) if users else None
        self.interference = interference  # "w" / "wo" / None

        self.held_out_user = held_out_user
        self.held_out_room = held_out_room
        self.held_out_ap = held_out_ap

        # find & parse files
        mat_paths = sorted(glob.glob(os.path.join(data_root, "**/*.mat"), recursive=True))
        records = []
        for p in mat_paths:
            meta = parse_rloc_filename(p)
            if meta is None:
                continue
            # apply filters (README semantics)
            if self.rooms is not None and meta["room"] not in self.rooms:
                continue
            if self.aps is not None and meta["ap"] not in self.aps:
                continue
            if self.users is not None and meta["user"] not in self.users:
                continue
            if self.interference is not None and meta["intf"] != self.interference:
                continue
            records.append((p, meta))

        if len(records) == 0:
            raise FileNotFoundError("No matched .mat files. Check filters / naming / data_root.")

        # build per-sample index (file-level expansion)
        all_items = []
        self._cache = {}
        for p, meta in records:
            md = mat73.loadmat(p)
            if "features_csi" not in md or "labels_aoa" not in md:
                continue

            csi = _safe_squeeze(md["features_csi"])
            y = _safe_squeeze(md["labels_aoa"]).reshape(-1)

            # Determine N packets
            # Possible: (N,90) or (90,N) or (90,) or (1,90)
            if csi.ndim == 1:
                # could be (90,) or (N*90,)
                if csi.shape[0] == 90:
                    N = 1
                else:
                    # if stored as flat array, try reshape
                    if csi.shape[0] % 90 != 0:
                        continue
                    N = csi.shape[0] // 90
            elif csi.ndim == 2:
                if csi.shape[1] == 90:
                    N = csi.shape[0]
                elif csi.shape[0] == 90:
                    N = csi.shape[1]
                else:
                    continue
            else:
                continue

            # align y length
            if y.size < N:
                continue
            if y.size > N:
                y = y[:N]

            for i in range(N):
                all_items.append((p, i, meta))

        if len(all_items) == 0:
            raise RuntimeError("No usable samples built from .mat files (check variable shapes).")

        # apply split strategy (README dimensions: room/ap/user/intf)
        self.index = self._split_items(all_items)

    def _split_items(self, all_items):
        rng = np.random.RandomState(self.seed)

        if self.split_strategy == "random":
            perm = rng.permutation(len(all_items))
            cut = int(len(all_items) * self.train_ratio)
            if self.split == "train":
                sel = perm[:cut]
            else:
                sel = perm[cut:]
            return [all_items[i] for i in sel]

        if self.split_strategy == "leave_one_user_out":
            if self.held_out_user is None:
                raise ValueError("held_out_user must be set for leave_one_user_out")
            if self.split == "train":
                return [it for it in all_items if it[2]["user"] != self.held_out_user]
            else:
                return [it for it in all_items if it[2]["user"] == self.held_out_user]

        if self.split_strategy == "cross_room":
            if self.held_out_room is None:
                raise ValueError("held_out_room must be set for cross_room")
            if self.split == "train":
                return [it for it in all_items if it[2]["room"] != self.held_out_room]
            else:
                return [it for it in all_items if it[2]["room"] == self.held_out_room]

        if self.split_strategy == "cross_ap":
            if self.held_out_ap is None:
                raise ValueError("held_out_ap must be set for cross_ap")
            if self.split == "train":
                return [it for it in all_items if it[2]["ap"] != self.held_out_ap]
            else:
                return [it for it in all_items if it[2]["ap"] == self.held_out_ap]

        raise ValueError(f"Unknown split_strategy: {self.split_strategy}")

    def __len__(self):
        return len(self.index)

    def _load_file(self, mat_path):
        if mat_path not in self._cache:
            self._cache[mat_path] = mat73.loadmat(mat_path)
        return self._cache[mat_path]

    def _get_packet(self, md, idx):
        csi = _safe_squeeze(md["features_csi"])
        y = _safe_squeeze(md["labels_aoa"]).reshape(-1)

        # CSI packet extraction
        if csi.ndim == 1:
            if csi.shape[0] == 90:
                csi_i = csi
            else:
                csi_i = csi[idx * 90:(idx + 1) * 90]
        else:
            if csi.shape[1] == 90:
                csi_i = csi[idx, :]
            else:
                csi_i = csi[:, idx]

        y_i = float(y[idx])

        # optional RSSI/AGC (README: rssi 1x3, agc scalar/1x1)
        rssi_i = None
        agc_i = None
        if self.include_rssi_agc:
            if "features_rssi" in md:
                rssi = _safe_squeeze(md["features_rssi"])
                # could be (N,3) / (3,N) / (3,)
                if rssi.ndim == 1 and rssi.shape[0] == 3:
                    rssi_i = rssi
                elif rssi.ndim == 2:
                    if rssi.shape[1] == 3:
                        rssi_i = rssi[idx, :]
                    elif rssi.shape[0] == 3:
                        rssi_i = rssi[:, idx]
            if "features_agc" in md:
                agc = _safe_squeeze(md["features_agc"])
                # could be (N,) or scalar
                if np.isscalar(agc) or agc.ndim == 0:
                    agc_i = float(agc)
                else:
                    agc_i = float(np.asarray(agc).reshape(-1)[idx])
                    
        # baseline: estimations_aoa from 2D-FFT
        est_i = None
        if "estimations_aoa" in md:
            est = _safe_squeeze(md["estimations_aoa"])
            est = np.asarray(est).reshape(-1)

            # common: (N,) or (1,N) or (N,1)
            if est.size > idx:
                est_i = float(est[idx])
            elif est.size > 0:
                # fallback if only one value
                est_i = float(est[0])

        return csi_i, y_i, est_i, rssi_i, agc_i

    def __getitem__(self, i):
        mat_path, idx, meta = self.index[i]
        md = self._load_file(mat_path)
        csi_i, y_i, est_i, rssi_i, agc_i = self._get_packet(md, idx)

        # Build complex AoA-ToF map + beamwidth
        X, beamwidth = build_aoa_tof_map_from_csi(csi_i, gA=self.gA, gD=self.gD)

        re = np.real(X)[None, :, :]  # (1,gA,gD)
        im = np.imag(X)[None, :, :]  # (1,gA,gD)
        bw = (beamwidth / self.gA)
        bw_map = np.ones((1, self.gA, self.gD), dtype=np.float32) * bw

        chans = [re, im, bw_map]

        # Optional: inject RSSI/AGC as constant maps (simple but effective baseline)
        # RSSI is 3 values (one per antenna). We map mean RSSI to a single channel.
        if self.include_rssi_agc:
            if rssi_i is not None:
                rssi_mean = float(np.mean(rssi_i))
            else:
                rssi_mean = 0.0
            rssi_map = np.ones((1, self.gA, self.gD), dtype=np.float32) * rssi_mean

            if agc_i is not None:
                agc_val = float(agc_i)
            else:
                agc_val = 0.0
            agc_map = np.ones((1, self.gA, self.gD), dtype=np.float32) * agc_val

            chans.extend([rssi_map, agc_map])

        inp = np.concatenate(chans, axis=0).astype(np.float32)  # (C,gA,gD)

        if self.angle_unit == "deg":
            target = np.float32(y_i)
        else:
            target = np.float32(np.deg2rad(y_i))

        # baseline AoA (deg/rad aligned)
        if est_i is None:
            base = np.float32(np.nan)
        else:
            base = np.float32(est_i if self.angle_unit == "deg" else np.deg2rad(est_i))
            
        # Also return meta if you want debugging / analysis
        return torch.from_numpy(inp), torch.tensor(target, dtype=torch.float32), torch.tensor(base), meta


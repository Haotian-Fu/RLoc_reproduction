import os
import math
import time
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from tqdm import tqdm

from dataset import HWILDDataset
from model import RLocUNetGaussian, gaussian_nll_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def make_logger(log_path):
    """
    Returns a function log(msg) that prints and appends msg to log_path.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log

def summarize_meta_batch(meta):
    """
    meta is usually a dict-of-lists after DataLoader collate.
    Return Counter stats for keys: room, ap, user, intf
    """
    stats = {}
    if isinstance(meta, dict):
        for k in ["room", "ap", "user", "intf"]:
            if k in meta:
                v = meta[k]
                # v could be list[str], list[int], or tensor
                if torch.is_tensor(v):
                    v = v.detach().cpu().tolist()
                stats[k] = Counter(v)
    return stats

def merge_counters(dst, src):
    """
    dst/src are dict[str, Counter]
    """
    for k, c in src.items():
        if k not in dst:
            dst[k] = Counter()
        dst[k].update(c)

@torch.no_grad()
def eval_epoch(model, loader, device, log=None, debug_meta=False):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    
    # accumulate for diagnostics
    all_abs_err = []
    all_sq_err = []

    base_abs_err = []
    base_sq_err = []
    base_n = 0
    
    meta_epoch = {}  # dict[str, Counter]
    
    all_mu = []
    all_sigma = []
    for x, y, base, meta in loader:
        x = x.to(device)
        y = y.to(device)
        base = base.to(device, non_blocking=True)
        mu, logvar = model(x)
        sigma = torch.exp(0.5 * logvar)
        all_mu.append(mu.detach().cpu())
        all_sigma.append(sigma.detach().cpu())
        loss = gaussian_nll_loss(y, mu, logvar).mean()

        # mae = torch.mean(torch.abs(y - mu))
        bs = x.size(0)
        total_loss += loss.item() * bs
        # total_mae += mae.item() * bs
        n += bs
        
        # # ----- (1) sigma stats -----
        # sigma = torch.exp(0.5 * logvar)  # (B,)
        # all_sigma.append(sigma.detach().cpu())

        # ----- (2) model error stats -----
        err = (y - mu)
        abs_err = torch.abs(err)
        all_abs_err.append(abs_err.detach().cpu())
        all_sq_err.append((err ** 2).detach().cpu())

        # ----- (3) baseline error stats -----
        # base can be NaN if missing; ignore NaNs
        valid = torch.isfinite(base)
        if valid.any():
            b_err = (y[valid] - base[valid])
            b_abs = torch.abs(b_err).detach().cpu()
            base_abs_err.append(b_abs)
            base_sq_err.append((b_err ** 2).detach().cpu())
            base_n += int(valid.sum().item())
        
        if debug_meta:
            merge_counters(meta_epoch, summarize_meta_batch(meta))
        
    # -------- reduce tensors --------
    def _cat(xs):
        if len(xs) == 0:
            return None
        return torch.cat(xs, dim=0)

    abs_err_all = _cat(all_abs_err)
    sq_err_all = _cat(all_sq_err)
    sigma_all = _cat(all_sigma)

    base_abs_all = _cat(base_abs_err)
    base_sq_all = _cat(base_sq_err)

    avg_loss = total_loss / max(n, 1)
    mu_all = torch.cat(all_mu, dim=0)
    sigma_all = torch.cat(all_sigma, dim=0)

    def stat_tensor(t):
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "p50": float(torch.quantile(t, 0.50).item()),
            "p90": float(torch.quantile(t, 0.90).item()),
            "p99": float(torch.quantile(t, 0.99).item()),
        }

    mu_stat = stat_tensor(mu_all)
    sigma_stat = stat_tensor(sigma_all)

    # helper for percentiles
    def _pct(t, ps=(50, 90, 99)):
        # t is 1D CPU tensor
        out = {}
        if t is None or t.numel() == 0:
            for p in ps:
                out[f"p{p}"] = float("nan")
            return out
        qt = torch.quantile(t, torch.tensor([p/100 for p in ps]))
        for i, p in enumerate(ps):
            out[f"p{p}"] = float(qt[i].item())
        return out

    # model metrics
    mae = float(abs_err_all.mean().item())
    rmse = float(torch.sqrt(sq_err_all.mean()).item())
    abs_pcts = _pct(abs_err_all, ps=(50, 90, 99))

    # sigma metrics
    sigma_mean = float(sigma_all.mean().item())
    sigma_pcts = _pct(sigma_all, ps=(50, 90, 99))

    # baseline metrics
    if base_abs_all is None or base_abs_all.numel() == 0:
        b_mae = float("nan")
        b_rmse = float("nan")
        b_pcts = {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    else:
        b_mae = float(base_abs_all.mean().item())
        b_rmse = float(torch.sqrt(base_sq_all.mean()).item())
        b_pcts = _pct(base_abs_all, ps=(50, 90, 99))

    # ---- logging block (CLI + txt) ----
    if log is not None:
        log(f"[VAL] loss={avg_loss:.6f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"abs_err(p50/p90/p99)={abs_pcts['p50']:.3f}/{abs_pcts['p90']:.3f}/{abs_pcts['p99']:.3f}")
        log(f"[VAL][mu] mean={mu_stat['mean']:.3f}, std={mu_stat['std']:.3f}, "
            f"min={mu_stat['min']:.3f}, max={mu_stat['max']:.3f}, "
            f"p50={mu_stat['p50']:.3f}, p90={mu_stat['p90']:.3f}, p99={mu_stat['p99']:.3f}")
        log(f"[VAL][sigma] mean={sigma_stat['mean']:.3f}, std={sigma_stat['std']:.3f}, "
            f"min={sigma_stat['min']:.3f}, max={sigma_stat['max']:.3f}, "
            f"p50={sigma_stat['p50']:.3f}, p90={sigma_stat['p90']:.3f}, p99={sigma_stat['p99']:.3f}")
        log(f"[VAL][baseline 2D-FFT] n_valid={base_n}/{n}  "
            f"MAE={b_mae:.3f}  RMSE={b_rmse:.3f}  "
            f"abs_err(p50/p90/p99)={b_pcts['p50']:.3f}/{b_pcts['p90']:.3f}/{b_pcts['p99']:.3f}")

        if debug_meta:
            log(f"[VAL meta] { {k: dict(v) for k,v in meta_epoch.items()} }")

    # return structured metrics to caller
    metrics = {
        "loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "abs_p50": abs_pcts["p50"],
        "abs_p90": abs_pcts["p90"],
        "abs_p99": abs_pcts["p99"],
        "mu_mean": mu_stat["mean"],
        "mu_p90": mu_stat["p90"],
        "sigma_mean": sigma_mean,
        "sigma_p50": sigma_pcts["p50"],
        "sigma_p90": sigma_pcts["p90"],
        "sigma_p99": sigma_pcts["p99"],
        "baseline_n": base_n,
        "baseline_mae": b_mae,
        "baseline_rmse": b_rmse,
        "baseline_abs_p50": b_pcts["p50"],
        "baseline_abs_p90": b_pcts["p90"],
        "baseline_abs_p99": b_pcts["p99"],
    }
    return metrics

def train_one_model(
    data_root,
    output_dir,
    seed=0,
    gA=64,
    gD=64,
    angle_unit="deg",
    batch_size=64,
    lr=1e-3,
    epochs=20,
    num_workers=4,
    device="cuda",
    debug_meta=True,
    include_rssi_agc=False
):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"train_log.txt")
    log = make_logger(log_path)

    # 建议把本次训练 config 也写下来，方便复现
    cfg = dict(
        data_root=data_root, seed=seed, gA=gA, gD=gD, angle_unit=angle_unit,
        batch_size=batch_size, lr=lr, epochs=epochs, num_workers=num_workers,
        device=device, debug_meta=debug_meta, include_rssi_agc=include_rssi_agc,
        split_strategy="leave_one_user_out", held_out_user=3,
    )
    with open(os.path.join(output_dir, f"config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    set_seed(seed)

    train_set = HWILDDataset(data_root, split="train",
                            split_strategy="leave_one_user_out",
                            held_out_user=3)
    val_set   = HWILDDataset(data_root, split="val",
                            split_strategy="leave_one_user_out",
                            held_out_user=3)


    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader   = DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    in_ch = 5 if include_rssi_agc else 3
    model = RLocUNetGaussian(in_ch=in_ch, base=64, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_path = os.path.join(output_dir, f"rloc_seed{seed}.pt")
    
    log(f"=== Start training seed={seed} ===")
    log(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    log(f"Log file: {log_path}")
    log(f"Best ckpt: {best_path}")

    for ep in range(1, epochs + 1):
        model.train()
        meta_epoch = {}  # dict[str, Counter]
        
        pbar = tqdm(train_loader, desc=f"[seed={seed}] epoch {ep}/{epochs}")
        running = 0.0
        steps = 0
        
        for x, y, base, meta in pbar:
            x = x.to(device)
            y = y.to(device)

            mu, logvar = model(x)
            loss = gaussian_nll_loss(y, mu, logvar).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=float(loss.item()))

            if debug_meta:
                merge_counters(meta_epoch, summarize_meta_batch(meta))

        sch.step()

        train_loss = running / max(steps, 1)
        val_metrics = eval_epoch(model, val_loader, device, log=log, debug_meta=debug_meta)

        val_loss = val_metrics["loss"]
        val_mae  = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]

        # 这里 log() 会同时 CLI 打印 + 写 txt，所以可以不用额外 print()
        log(f"[epoch {ep}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f} "
            f"mu_mean={val_metrics['mu_mean']:.3f} "
            f"sigma_mean={val_metrics['sigma_mean']:.3f} sigma_p90={val_metrics['sigma_p90']:.3f} "
            f"baseline_MAE={val_metrics['baseline_mae']:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "seed": seed,
                "gA": gA, "gD": gD,
                "angle_unit": angle_unit,
                "in_ch": in_ch,
                "best_val": best_val,
                "val_metrics": val_metrics,  # ✅ 把当时的诊断也存进 ckpt
            }, best_path)
            log(f"[epoch {ep}] ✅ New best val_loss={best_val:.6f}. Saved to {best_path}")

        if debug_meta:
            # 把 Counter 转成 dict 便于读
            meta_readable = {k: dict(v) for k, v in meta_epoch.items()}
            log(f"[TRAIN meta] {meta_readable}")

    log("=== Training done ===")
    return best_path

@torch.no_grad()
def ensemble_predict(models, x):
    """
    models: list of (model) each outputs (mu, logvar)
    x: (B,C,H,W)
    Returns:
      mu_hat: (B,)
      var_hat: (B,)   total variance = epistemic + aleatoric
    """
    mus = []
    vars_ = []
    for m in models:
        mu, logvar = m(x)
        mus.append(mu)
        vars_.append(torch.exp(logvar))  # aleatoric var

    mus = torch.stack(mus, dim=0)    # (Z,B)
    vars_ = torch.stack(vars_, dim=0) # (Z,B)

    mu_hat = mus.mean(dim=0)  # (B,)
    epistemic = mus.var(dim=0, unbiased=False)  # (B,)
    aleatoric = vars_.mean(dim=0)  # (B,)
    var_hat = epistemic + aleatoric
    return mu_hat, var_hat

def train_deep_ensemble(data_root, output_dir, seeds=(0,1,2,3,4), **kwargs):
    paths = []
    for s in seeds:
        seed_dir = os.path.join(output_dir, f"seed{s}")
        p = train_one_model(
            data_root=data_root,
            output_dir=seed_dir,
            seed=s,
            **kwargs
        )
        paths.append(p)
    return paths

if __name__ == "__main__":
    # ====== YOU CHANGE THESE ======
    DATA_ROOT = "/home/haotian/RLoc/dataset/human_held_device_wifi_indoor_localization_dataset-main/Lounge"
    OUTPUT_DIR = "/home/haotian/RLoc/reproduction/results/S6/leave_one_user_out/epoch100"
    SEED = 42
    # ==============================

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, e.g., 0/1/2/3")
    parser.add_argument("--seed", type=int, default=None, help="single seed")
    parser.add_argument("--seeds", type=str, default=None, help="comma list, e.g. 0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--include_rssi_agc", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"

    # decide seeds
    if args.seeds is not None:
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    elif args.seed is not None:
        seeds = (int(args.seed),)
    else:
        raise ValueError("Please set --seed or --seeds")

    # 如果你在“每卡一个seed并行”，通常每个进程只传一个 seed
    # 但这里也允许一个进程顺序跑多个 seed（不并行）
    if len(seeds) == 1:
        s = seeds[0]
        seed_dir = os.path.join(args.output_dir, f"seed{s}")
        best = train_one_model(
            data_root=args.data_root,
            output_dir=seed_dir,
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=device,
            angle_unit="deg",
            include_rssi_agc=args.include_rssi_agc,
        )
        print("Saved:", best)
    else:
        paths = train_deep_ensemble(
            args.data_root,
            args.output_dir,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=device,
            angle_unit="deg",
            include_rssi_agc=args.include_rssi_agc,
        )
        print("Ensemble saved:", paths)
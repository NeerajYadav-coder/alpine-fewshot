# run_falsification_exp4_protonet.py
#
# FALSIFICATION EXPERIMENT 4: MODERNIZED PROTONET BASELINE
# Tests whether adding LayerNorm and switching to AdamW (weight_decay=1e-4)
# on ProtoNet closes the accuracy gap to EXP-F3 (22k) and EXP-F3-35k (35k).
#
# Evaluates ProtoNet-Modernized (47,770 params) across seeds [1, 7, 21, 42, 123]
# on CIFAR-FS and MiniImageNet Native (1-shot and 5-shot).
# Saves checkpoints to checkpoints/sota_protonet_modernized_*.pt,
# performs reload verification, and generates checkpoints/falsification_protonet_modernized_manifest.json.

import datetime
import hashlib
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cifar_fs_dataset import RealCIFARFS
from mini_imagenet_mmap_loader import MiniImageNetMmap as RealMiniImageNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIVE_SEEDS    = [1, 7, 21, 42, 123]
META_TRAIN_EP = 250
EVAL_EPISODES = 600
CHECKPOINT_DIR = "checkpoints"
MANIFEST_PATH  = os.path.join(CHECKPOINT_DIR, "falsification_protonet_modernized_manifest.json")

# =====================================================================
# MODERNIZED PROTONET BASELINE (LayerNorm + AdamW)
# =====================================================================
class ProtoNetModernized(nn.Module):
    """ProtoNet (Snell et al. 2017) modernized with LayerNorm and matched output normalization."""
    def __init__(self, hidden=40, out_dim=70):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(hidden, out_dim)
        self.ln_out = nn.LayerNorm(out_dim)

    def extract(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return F.normalize(self.ln_out(self.fc(h)), p=2, dim=-1)

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

def train_protonet_modernized(ds_train, k_shot=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProtoNetModernized().to(device)
    # Modernized Optimizer: AdamW with weight_decay=1e-4 matching EXP-F3 exact optimizer
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.5)
    crit  = nn.CrossEntropyLoss()
    model.train()
    for ep in range(META_TRAIN_EP):
        opt.zero_grad()
        sx, sy, qx, qy, _ = ds_train.sample_episode(n_way=5, k_shot=k_shot, q_query=15, seed=seed+ep)
        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
        protos = model.compute_prototypes(sx, sy)
        loss   = crit(model.predict_proto(qx, protos), qy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
    return model

def eval_protonet_modernized(model, ds_test, k_shot=5, seed=42):
    model.eval()
    accs = []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=5, k_shot=k_shot, q_query=15, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            protos = model.compute_prototypes(sx, sy, n=5)
            preds  = model.predict_proto(qx, protos)
            accs.append((preds.argmax(1) == qy).float().mean().item() * 100.0)
    return float(np.mean(accs))

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def run_falsification_exp4():
    print("=" * 100)
    print("FALSIFICATION EXPERIMENT 4: MODERNIZED PROTONET BASELINE (LayerNorm + AdamW)")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 100)

    m = ProtoNetModernized()
    pc = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Parameter Audit: ProtoNet-Modernized = {pc:,} trainable parameters (Original ProtoNet = 47,630).", flush=True)

    cifar_train = RealCIFARFS(split="train"); cifar_test = RealCIFARFS(split="test")
    mini_train  = RealMiniImageNet(split="train"); mini_test = RealMiniImageNet(split="test")

    datasets = [
        ("cifar", cifar_train, cifar_test),
        ("mini",  mini_train,  mini_test)
    ]

    manifest_entries = []
    results = {}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for ds_name, ds_train, ds_test in datasets:
        results[ds_name] = {}
        for shot in [1, 5]:
            results[ds_name][shot] = {}
            print(f"\n==================================================================================", flush=True)
            print(f"RUNNING MATRIX: Dataset={ds_name.upper()} | Shot={shot}-Shot", flush=True)
            print(f"==================================================================================", flush=True)
            for seed in FIVE_SEEDS:
                ckpt_filename = f"sota_protonet_modernized_{ds_name}_{shot}shot_seed{seed}.pt"
                ckpt_path     = os.path.join(CHECKPOINT_DIR, ckpt_filename)

                # 1. Train
                print(f"[{ds_name.upper()} {shot}-Shot Seed {seed}] Training ProtoNet-Modernized...", end=" ", flush=True)
                model = train_protonet_modernized(ds_train, k_shot=shot, seed=seed)

                # 2. Evaluate
                acc_rep = eval_protonet_modernized(model, ds_test, k_shot=shot, seed=seed)
                print(f"Done. Reported Acc: {acc_rep:.2f}%", flush=True)
                results[ds_name][shot][seed] = acc_rep

                # 3. Save Checkpoint
                ckpt_dict = {
                    "model_state_dict": model.state_dict(),
                    "reported_accuracy": float(acc_rep),
                    "dataset": ds_name,
                    "shot": shot,
                    "seed": seed,
                    "trainable_parameters": pc
                }
                torch.save(ckpt_dict, ckpt_path)
                md5_val = compute_md5(ckpt_path)

                # 4. Reload Verification
                reload_model = ProtoNetModernized().to(device)
                loaded_dict  = torch.load(ckpt_path, map_location=device, weights_only=False)
                reload_model.load_state_dict(loaded_dict["model_state_dict"])
                acc_reload  = eval_protonet_modernized(reload_model, ds_test, k_shot=shot, seed=seed)

                verified = bool(abs(acc_rep - acc_reload) < 0.05)
                print(f"   ↳ Checkpoint Saved: {ckpt_filename} | Reload Acc: {acc_reload:.2f}% | MD5: {md5_val} | Verified: {verified}", flush=True)

                manifest_entries.append({
                    "filename": ckpt_filename,
                    "dataset": ds_name,
                    "shot": shot,
                    "seed": seed,
                    "reported_std_accuracy": round(float(acc_rep), 2),
                    "reloaded_std_accuracy": round(float(acc_reload), 4),
                    "md5": md5_val,
                    "verified": verified
                })

    # Build Master Manifest
    all_verif = all(e["verified"] for e in manifest_entries)
    manifest_data = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
        "model_class": "ProtoNetModernized (Conv-4 + LayerNorm + AdamW)",
        "trainable_parameters": pc,
        "total_checkpoints": len(manifest_entries),
        "all_verified": all_verif,
        "checkpoints": manifest_entries
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 100)
    print(f"FALSIFICATION EXP 4 COMPLETE. MANIFEST SAVED TO {MANIFEST_PATH}")
    print(f"All 20 Checkpoints Verified: {all_verif}")
    print("=" * 100)

    # Save summary results JSON
    summary_path = os.path.join(CHECKPOINT_DIR, "protonet_modernized_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_falsification_exp4()

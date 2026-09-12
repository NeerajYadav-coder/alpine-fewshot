# run_exp_f3_35k_matrix.py
#
# FULL 5-SEED, 2-DATASET, 2-SHOT VERIFICATION OF EXP-F3-35k (34,917 PARAMS)
# Evaluates EXP-F3-35k across seeds [1, 7, 21, 42, 123] on CIFAR-FS and MiniImageNet Native.
# Saves checkpoints, computes MD5 hashes, performs reload accuracy verification,
# and generates checkpoints/exp_f3_35k_manifest.json.

import datetime
import hashlib
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cifar_fs_dataset import RealCIFARFS
from mini_imagenet_mmap_loader import MiniImageNetMmap as RealMiniImageNet
from run_exp_f3_benchmark import get_exp_p1_base_centers, WideWindowAdaptivePatchLocator, extract_patches_grid_sample
from irfe_p1_canonical import GaborPreprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_EPISODES = 600
META_TRAIN_EP = 250
SEEDS = [1, 7, 21, 42, 123]
CHECKPOINT_DIR = "checkpoints"
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "exp_f3_35k_manifest.json")

class PatchEncoderCIFARGeneric(nn.Module):
    def __init__(self, c1, c2, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv1 = nn.Conv2d(7, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(c2 * 4 * 4, out_dim)
        self.ln    = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.ln(self.fc(x.view(x.size(0), -1)))

class SubPatchEncoderCIFARGeneric(nn.Module):
    def __init__(self, sub_c, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, sub_c, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(sub_c * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class BoundaryEncoderCIFARGeneric(nn.Module):
    def __init__(self, sub_c, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, sub_c, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(sub_c * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class PatchEncoderNativeGeneric(nn.Module):
    def __init__(self, c1, c2, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv1 = nn.Conv2d(7, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.adap  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(c2 * 4 * 4, out_dim)
        self.ln    = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.gabor(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adap(x)
        return self.ln(self.fc(x.view(x.size(0), -1)))

class SubPatchEncoderNativeGeneric(nn.Module):
    def __init__(self, sub_c, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, sub_c, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(sub_c * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

class BoundaryEncoderNativeGeneric(nn.Module):
    def __init__(self, sub_c, out_dim):
        super().__init__()
        self.gabor = GaborPreprocess(ksize=5)
        self.conv  = nn.Conv2d(7, sub_c, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(sub_c * 4 * 4, out_dim)

    def forward(self, x):
        x = self.gabor(x)
        return F.relu(self.fc(self.pool(F.relu(self.conv(x))).view(x.size(0), -1)))

# 35k Model CIFAR (34,917 params)
class IRFEExpF3_35k_CIFAR(nn.Module):
    def __init__(self, c1=15, c2=19, sub_c=8, gate_h=12, embed_dim=32, window_frac=0.50, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        base_centers          = get_exp_p1_base_centers()
        self.locator          = WideWindowAdaptivePatchLocator(base_centers, window_frac=window_frac)
        self.encoder          = PatchEncoderCIFARGeneric(c1, c2, out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderCIFARGeneric(sub_c, out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderCIFARGeneric(sub_c, out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_h), nn.ReLU(), nn.Linear(gate_h, embed_dim * 2)
        )

    def _split_boundaries(self, x):
        return (x[:,:, 8:16, 8:16], x[:,:, 8:16, 16:24],
                x[:,:, 16:24, 8:16], x[:,:, 16:24, 16:24])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+8, c:c+8] for r in [0, 8] for c in [0, 8]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=0.5)
        patches     = [self._encode_patch(p) for p in raw_patches]
        edges       = [self._rel(patches[i], patches[j]) for i in range(5) for j in range(i+1, 5)]
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)]
        
        tokens  = torch.stack(patches + edges + boundaries, dim=1)
        attn, _ = self.mha(tokens, tokens, tokens)
        refined = self.ln_attn(tokens + attn)
        ws      = self.ws_query.expand(B, -1, -1)
        scores  = torch.bmm(ws, refined.transpose(1, 2)) / math.sqrt(self.embed_dim)
        W       = torch.bmm(F.softmax(scores, dim=-1), refined).squeeze(1)
        
        patch_pool = refined[:, :5, :].mean(dim=1)
        g       = torch.cat([patch_pool, W], dim=1)
        gate    = 2.0 * torch.sigmoid(self.gate_net(g)) - 1.0
        return g + g * gate

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

# 35k Model Native (34,917 params)
class IRFEExpF3_35k_Native(nn.Module):
    def __init__(self, c1=15, c2=19, sub_c=8, gate_h=12, embed_dim=32, window_frac=0.50, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        base_centers          = get_exp_p1_base_centers()
        self.locator          = WideWindowAdaptivePatchLocator(base_centers, window_frac=window_frac)
        self.encoder          = PatchEncoderNativeGeneric(c1, c2, out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderNativeGeneric(sub_c, out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderNativeGeneric(sub_c, out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_h), nn.ReLU(), nn.Linear(gate_h, embed_dim * 2)
        )

    def _split_boundaries(self, x):
        return (x[:,:, 18:48, 18:48], x[:,:, 18:48, 36:66],
                x[:,:, 36:66, 18:48], x[:,:, 36:66, 36:66])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+24, c:c+24] for r in [0, 24] for c in [0, 24]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=48.0/84.0)
        patches     = [self._encode_patch(p) for p in raw_patches]
        edges       = [self._rel(patches[i], patches[j]) for i in range(5) for j in range(i+1, 5)]
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)]
        
        tokens  = torch.stack(patches + edges + boundaries, dim=1)
        attn, _ = self.mha(tokens, tokens, tokens)
        refined = self.ln_attn(tokens + attn)
        ws      = self.ws_query.expand(B, -1, -1)
        scores  = torch.bmm(ws, refined.transpose(1, 2)) / math.sqrt(self.embed_dim)
        W       = torch.bmm(F.softmax(scores, dim=-1), refined).squeeze(1)
        
        patch_pool = refined[:, :5, :].mean(dim=1)
        g       = torch.cat([patch_pool, W], dim=1)
        gate    = 2.0 * torch.sigmoid(self.gate_net(g)) - 1.0
        return g + g * gate

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

# -----------------------------------------------------------------------------
# Training & Evaluation Protocol
# -----------------------------------------------------------------------------
def meta_train(model_cls, ds_train, k_shot, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls().to(device)
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

def eval_standard_600(model, ds_test, k_shot, seed):
    model.eval()
    accs = []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=5, k_shot=k_shot, q_query=15, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            protos = model.compute_prototypes(sx, sy, n=5)
            preds = model.predict_proto(qx, protos)
            accs.append((preds.argmax(1) == qy).float().mean().item() * 100.0)
    return np.mean(accs)

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def run_matrix():
    print("=" * 100)
    print("EXP-F3-35K (34,917 PARAMS) — FULL 5-SEED VERIFICATION MATRIX")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 100)

    mc = IRFEExpF3_35k_CIFAR()
    mn = IRFEExpF3_35k_Native()
    pc_c = sum(p.numel() for p in mc.parameters() if p.requires_grad)
    pc_n = sum(p.numel() for p in mn.parameters() if p.requires_grad)
    assert pc_c == 34917 and pc_n == 34917, f"Parameter mismatch! Got CIFAR={pc_c}, Native={pc_n}"
    print(f"Parameter Audit Confirmed: EXP-F3-35k (CIFAR & Native) = {pc_c:,} trainable parameters.", flush=True)

    cifar_train = RealCIFARFS(split="train"); cifar_test = RealCIFARFS(split="test")
    mini_train  = RealMiniImageNet(split="train"); mini_test = RealMiniImageNet(split="test")

    datasets = [
        ("cifar", IRFEExpF3_35k_CIFAR, cifar_train, cifar_test),
        ("mini",  IRFEExpF3_35k_Native, mini_train,  mini_test)
    ]

    manifest_entries = []
    results = {}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for ds_name, model_cls, ds_train, ds_test in datasets:
        results[ds_name] = {}
        for shot in [1, 5]:
            results[ds_name][shot] = {}
            print(f"\n==================================================================================", flush=True)
            print(f"RUNNING MATRIX: Dataset={ds_name.upper()} | Shot={shot}-Shot", flush=True)
            print(f"==================================================================================", flush=True)
            for seed in SEEDS:
                ckpt_filename = f"exp_f3_35k_{ds_name}_{shot}shot_seed{seed}.pt"
                ckpt_path     = os.path.join(CHECKPOINT_DIR, ckpt_filename)

                # 1. Train
                print(f"[{ds_name.upper()} {shot}-Shot Seed {seed}] Training...", end=" ", flush=True)
                model = meta_train(model_cls, ds_train, k_shot=shot, seed=seed)

                # 2. Evaluate
                acc_rep = eval_standard_600(model, ds_test, k_shot=shot, seed=seed)
                print(f"Done. Reported Acc: {acc_rep:.2f}%", flush=True)
                results[ds_name][shot][seed] = acc_rep

                # 3. Save Checkpoint
                ckpt_dict = {
                    "model_state_dict": model.state_dict(),
                    "reported_accuracy": acc_rep,
                    "dataset": ds_name,
                    "shot": shot,
                    "seed": seed,
                    "trainable_parameters": pc_c
                }
                torch.save(ckpt_dict, ckpt_path)
                md5_val = compute_md5(ckpt_path)

                # 4. Reload Verification
                reload_model = model_cls().to(device)
                loaded_dict  = torch.load(ckpt_path, map_location=device, weights_only=False)
                reload_model.load_state_dict(loaded_dict["model_state_dict"])
                acc_reload  = eval_standard_600(reload_model, ds_test, k_shot=shot, seed=seed)

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
        "model_class": "IRFEExpF3_35k (WindowedAdaptivePatchLocator, window_frac=0.50)",
        "trainable_parameters": 34917,
        "total_checkpoints": len(manifest_entries),
        "all_verified": all_verif,
        "checkpoints": manifest_entries
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 100)
    print(f"EXP-F3-35K VERIFICATION MATRIX COMPLETE. MANIFEST SAVED TO {MANIFEST_PATH}")
    print(f"All 20 Checkpoints Verified: {all_verif}")
    print("=" * 100)

    # Save summary results JSON for easy reporting
    summary_path = os.path.join(CHECKPOINT_DIR, "exp_f3_35k_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_matrix()

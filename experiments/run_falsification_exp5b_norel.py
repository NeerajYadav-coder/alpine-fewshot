# run_falsification_exp5b_norel.py
#
# FALSIFICATION EXPERIMENT 5b: TRAIN-FROM-SCRATCH WITHOUT RELATIONAL TOKENS (TRUE CAUSAL TEST)
# Tests whether relational tokens were causally necessary DURING learning.
#
# Physically removes the 10 pairwise relational tokens (e_ij) and rel_proj module.
# Token sequence fed into MHA becomes 9 tokens (5 patches + 4 boundary crops).
# Trains fresh models from scratch (250 meta-training episodes, 600 eval episodes)
# across seeds [1, 7, 21, 42, 123] on CIFAR-FS and MiniImageNet Native (1-shot and 5-shot).
#
# Model Variants:
# 1. EXP-F3-NoRel     : 21,689 trainable parameters (base 22,249 - 560 params)
# 2. EXP-F3-35k-NoRel : 32,773 trainable parameters (base 34,917 - 2,144 params)

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
from run_exp_f3_benchmark import (
    PatchEncoderCIFARP1,
    SubPatchEncoderCIFARP1,
    BoundaryEncoderCIFARP1,
    PatchEncoderNativeP1,
    SubPatchEncoderNativeP1,
    BoundaryEncoderNativeP1,
    get_exp_p1_base_centers,
    WideWindowAdaptivePatchLocator,
    extract_patches_grid_sample
)
from run_30k_35k_probes import (
    PatchEncoderCIFARGeneric,
    SubPatchEncoderCIFARGeneric,
    BoundaryEncoderCIFARGeneric,
    PatchEncoderNativeGeneric,
    SubPatchEncoderNativeGeneric,
    BoundaryEncoderNativeGeneric
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIVE_SEEDS    = [1, 7, 21, 42, 123]
META_TRAIN_EP = 250
EVAL_EPISODES = 600
CHECKPOINT_DIR = "checkpoints"
MANIFEST_PATH  = os.path.join(CHECKPOINT_DIR, "falsification_exp5b_manifest.json")

# =====================================================================
# 1. EXP-F3-NoRel (21,689 params) — CIFAR & Native
# =====================================================================
class IRFEExpF3_NoRel_CIFAR(nn.Module):
    def __init__(self, embed_dim=16, window_frac=0.50, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        base_centers          = get_exp_p1_base_centers()
        self.locator          = WideWindowAdaptivePatchLocator(base_centers, window_frac=window_frac)
        self.encoder          = PatchEncoderCIFARP1(out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderCIFARP1(out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderCIFARP1(out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, embed_dim * 2)
        )

    def _split_boundaries(self, x):
        return (x[:,:, 8:16, 8:16], x[:,:, 8:16, 16:24],
                x[:,:, 16:24, 8:16], x[:,:, 16:24, 16:24])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+8, c:c+8] for r in [0, 8] for c in [0, 8]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=0.5)
        patches     = [self._encode_patch(p) for p in raw_patches] # 5 tokens
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)] # 4 tokens
        
        # 9 tokens total (NO relational edge tokens)
        tokens  = torch.stack(patches + boundaries, dim=1)
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

class IRFEExpF3_NoRel_Native(nn.Module):
    def __init__(self, embed_dim=16, window_frac=0.50, num_heads=4):
        super().__init__()
        self.embed_dim        = embed_dim
        base_centers          = get_exp_p1_base_centers()
        self.locator          = WideWindowAdaptivePatchLocator(base_centers, window_frac=window_frac)
        self.encoder          = PatchEncoderNativeP1(out_dim=embed_dim)
        self.sub_encoder      = SubPatchEncoderNativeP1(out_dim=embed_dim)
        self.boundary_encoder = BoundaryEncoderNativeP1(out_dim=embed_dim)
        self.sub_fusion       = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )
        self.mha          = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_attn      = nn.LayerNorm(embed_dim)
        self.ws_query     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gate_net     = nn.Sequential(
            nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, embed_dim * 2)
        )

    def _split_boundaries(self, x):
        return (x[:,:, 18:48, 18:48], x[:,:, 18:48, 36:66],
                x[:,:, 36:66, 18:48], x[:,:, 36:66, 36:66])

    def _encode_patch(self, p):
        v_c = self.encoder(p)
        sp = [p[:,:, r:r+24, c:c+24] for r in [0, 24] for c in [0, 24]]
        v_f = self.sub_fusion(torch.cat([self.sub_encoder(s) for s in sp], dim=1))
        return v_c + v_f

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=48.0/84.0)
        patches     = [self._encode_patch(p) for p in raw_patches] # 5 tokens
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)] # 4 tokens
        
        # 9 tokens total (NO relational edge tokens)
        tokens  = torch.stack(patches + boundaries, dim=1)
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

# =====================================================================
# 2. EXP-F3-35k-NoRel (32,773 params) — CIFAR & Native
# =====================================================================
class IRFEExpF3_35k_NoRel_CIFAR(nn.Module):
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

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=0.5)
        patches     = [self._encode_patch(p) for p in raw_patches] # 5 tokens
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)] # 4 tokens
        
        # 9 tokens total (NO relational edge tokens)
        tokens  = torch.stack(patches + boundaries, dim=1)
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

class IRFEExpF3_35k_NoRel_Native(nn.Module):
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

    def extract(self, x):
        B = x.size(0)
        gabor_out = self.encoder.gabor(x)
        energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
        centers    = self.locator(energy_map)

        raw_patches = extract_patches_grid_sample(x, centers, patch_scale=48.0/84.0)
        patches     = [self._encode_patch(p) for p in raw_patches] # 5 tokens
        boundaries  = [self.boundary_encoder(b) for b in self._split_boundaries(x)] # 4 tokens
        
        # 9 tokens total (NO relational edge tokens)
        tokens  = torch.stack(patches + boundaries, dim=1)
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

def run_falsification_exp5b():
    print("=" * 100)
    print("FALSIFICATION EXPERIMENT 5b: TRAIN-FROM-SCRATCH WITHOUT RELATIONAL TOKENS (TRUE CAUSAL TEST)")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 100)

    # Parameter Audit
    m22 = IRFEExpF3_NoRel_CIFAR(); pc22 = sum(p.numel() for p in m22.parameters() if p.requires_grad)
    m35 = IRFEExpF3_35k_NoRel_CIFAR(); pc35 = sum(p.numel() for p in m35.parameters() if p.requires_grad)

    print(f"Parameter Audit: EXP-F3-NoRel = {pc22:,} params (vs EXP-F3 22,249)", flush=True)
    print(f"Parameter Audit: EXP-F3-35k-NoRel = {pc35:,} params (vs EXP-F3-35k 34,917)", flush=True)

    cifar_train = RealCIFARFS(split="train"); cifar_test = RealCIFARFS(split="test")
    mini_train  = RealMiniImageNet(split="train"); mini_test = RealMiniImageNet(split="test")

    model_groups = [
        ("EXP-F3-NoRel", pc22, "exp_f3_norel", {
            "cifar": (IRFEExpF3_NoRel_CIFAR, cifar_train, cifar_test),
            "mini":  (IRFEExpF3_NoRel_Native, mini_train, mini_test)
        }),
        ("EXP-F3-35k-NoRel", pc35, "exp_f3_35k_norel", {
            "cifar": (IRFEExpF3_35k_NoRel_CIFAR, cifar_train, cifar_test),
            "mini":  (IRFEExpF3_35k_NoRel_Native, mini_train, mini_test)
        })
    ]

    manifest_entries = []
    results = {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for group_name, pc, prefix, ds_dict in model_groups:
        results[group_name] = {}
        print(f"\n==================================================================================", flush=True)
        print(f"TRAINING MODEL GROUP: {group_name} ({pc:,} params)", flush=True)
        print(f"==================================================================================", flush=True)

        for ds_name, (model_cls, ds_tr, ds_te) in ds_dict.items():
            results[group_name][ds_name] = {}
            for shot in [1, 5]:
                results[group_name][ds_name][shot] = {}
                print(f"\n--- Dataset={ds_name.upper()} | Shot={shot}-Shot ---", flush=True)

                for seed in FIVE_SEEDS:
                    ckpt_filename = f"{prefix}_{ds_name}_{shot}shot_seed{seed}.pt"
                    ckpt_path     = os.path.join(CHECKPOINT_DIR, ckpt_filename)

                    # 1. Train fresh from scratch
                    print(f"[{ds_name.upper()} {shot}-Shot Seed {seed}] Training {group_name}...", end=" ", flush=True)
                    model = meta_train(model_cls, ds_tr, k_shot=shot, seed=seed)

                    # 2. Evaluate
                    acc_rep = eval_standard_600(model, ds_te, k_shot=shot, seed=seed)
                    print(f"Done. Reported Acc: {acc_rep:.2f}%", flush=True)
                    results[group_name][ds_name][shot][seed] = float(acc_rep)

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
                    reload_model = model_cls().to(device)
                    loaded_dict  = torch.load(ckpt_path, map_location=device, weights_only=False)
                    reload_model.load_state_dict(loaded_dict["model_state_dict"])
                    acc_reload  = eval_standard_600(reload_model, ds_te, k_shot=shot, seed=seed)

                    verified = bool(abs(acc_rep - acc_reload) < 0.05)
                    print(f"   ↳ Checkpoint Saved: {ckpt_filename} | Reload Acc: {acc_reload:.2f}% | MD5: {md5_val} | Verified: {verified}", flush=True)

                    manifest_entries.append({
                        "group_name": group_name,
                        "filename": ckpt_filename,
                        "dataset": ds_name,
                        "shot": shot,
                        "seed": seed,
                        "reported_std_accuracy": round(float(acc_rep), 2),
                        "reloaded_std_accuracy": round(float(acc_reload), 4),
                        "md5": md5_val,
                        "verified": verified
                    })

    # Save manifest
    all_verif = all(e["verified"] for e in manifest_entries)
    manifest_data = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
        "experiment": "Falsification Experiment 5b: Train-From-Scratch WITHOUT Relational Tokens",
        "total_checkpoints": len(manifest_entries),
        "all_verified": all_verif,
        "checkpoints": manifest_entries
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 100)
    print(f"FALSIFICATION EXP 5b MATRIX COMPLETE. MANIFEST SAVED TO {MANIFEST_PATH}")
    print(f"All 40 Checkpoints Verified: {all_verif}")
    print("=" * 100)

    # Save summary results JSON
    summary_path = os.path.join(CHECKPOINT_DIR, "exp5b_norel_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_falsification_exp5b()

# run_exp_f3_benchmark.py
# EXP-F3 BENCHMARK: Windowed Adaptive Locator (window_frac = 0.50)
#
# Protocol:
# 1. Standard 5 Seeds: [1, 7, 21, 42, 123], 600 eval episodes per seed.
# 2. Datasets: CIFAR-FS (32x32) & MiniImageNet Native (84x84).
# 3. Shots: 1-Shot and 5-Shot.
# 4. Evaluation Conditions (Query Images):
#    - Condition 1: Standard (Unmodified)
#    - Condition 2: Partial-Object Robustness (50% random half-masking)
#    - Condition 3: Position-Shift Robustness (25% translation with zero-padding)
# 5. Parameter Accounting: 22,249 parameters (Identical to EXP-F1, +1 scalar for locator temperature).

import datetime
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
from irfe_p1_canonical import (
    GaborPreprocess,
    PatchEncoderCIFARP1,
    SubPatchEncoderCIFARP1,
    BoundaryEncoderCIFARP1,
    PatchEncoderNativeP1,
    SubPatchEncoderNativeP1,
    BoundaryEncoderNativeP1,
    IRFEExpP1CIFAR,
    IRFEExpP1Native
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIVE_SEEDS    = [1, 7, 21, 42, 123]
META_TRAIN_EP = 250
EVAL_EPISODES = 600

def get_exp_p1_base_centers():
    return torch.tensor([
        [-0.5161, -0.5161],
        [-0.5161,  0.5161],
        [ 0.0000,  0.0000],
        [ 0.5161, -0.5161],
        [ 0.5161,  0.5161]
    ], dtype=torch.float32)

class WideWindowAdaptivePatchLocator(nn.Module):
    def __init__(self, base_centers, window_frac=0.50, temperature=1.0):
        super().__init__()
        self.register_buffer("base_centers", base_centers)
        self.num_patches = base_centers.shape[0]
        self.window_frac = window_frac
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, energy_map):
        B, _, H, W = energy_map.shape
        ys = torch.linspace(-1, 1, H, device=energy_map.device)
        xs = torch.linspace(-1, 1, W, device=energy_map.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')

        flat_energy = energy_map.squeeze(1)

        centers = []
        for k in range(self.num_patches):
            cy0 = self.base_centers[k, 0]
            cx0 = self.base_centers[k, 1]
            
            dist_sq = (gy - cy0)**2 + (gx - cx0)**2
            log_window = -dist_sq / (2 * (self.window_frac**2) + 1e-6)
            
            weighted_energy = flat_energy / (self.temperature.abs() + 1e-4) + log_window
            w = F.softmax(weighted_energy.view(B, -1), dim=-1)
            cy = (w * gy.reshape(-1)).sum(-1)
            cx = (w * gx.reshape(-1)).sum(-1)
            centers.append(torch.stack([cx, cy], dim=-1))
        return torch.stack(centers, dim=1) # (B, 5, 2)

def extract_patches_grid_sample(x, centers, patch_scale=0.5):
    B, C, H, W = x.shape
    patch_h = int(round(H * patch_scale))
    patch_w = int(round(W * patch_scale))

    patches = []
    for k in range(centers.size(1)):
        cx = centers[:, k, 0]
        cy = centers[:, k, 1]
        
        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = patch_scale
        theta[:, 1, 1] = patch_scale
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy

        grid = F.affine_grid(theta, torch.Size([B, C, patch_h, patch_w]), align_corners=False)
        p = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
        patches.append(p)
    return patches

class IRFEExpF3_CIFAR(nn.Module):
    def __init__(self, window_frac=0.50, embed_dim=16, num_heads=4):
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
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
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

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract_with_rel_tokens(self, x):
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
        out_feat= g + g * gate

        rel_tokens = torch.stack(edges, dim=1)
        return out_feat, centers, rel_tokens

    def extract(self, x):
        feat, _, _ = self.extract_with_rel_tokens(x)
        return feat

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

class IRFEExpF3_Native(nn.Module):
    def __init__(self, window_frac=0.50, embed_dim=16, num_heads=4):
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
        self.rel_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
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

    def _rel(self, va, vb):
        return self.rel_proj(torch.cat([va - vb, va * vb], dim=1))

    def extract_with_rel_tokens(self, x):
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
        out_feat= g + g * gate

        rel_tokens = torch.stack(edges, dim=1)
        return out_feat, centers, rel_tokens

    def extract(self, x):
        feat, _, _ = self.extract_with_rel_tokens(x)
        return feat

    def compute_prototypes(self, sx, sy, n=5):
        f = self.extract(sx)
        return torch.stack([f[sy == c].mean(0) for c in range(n)])

    def predict_proto(self, qx, protos):
        return -(torch.cdist(self.extract(qx), protos) ** 2)

# =====================================================================
# EVALUATION CONDITIONS
# =====================================================================
def apply_partial_mask(qx, seed=42):
    B, C, H, W = qx.shape
    g = torch.Generator()
    g.manual_seed(seed)
    qx_masked = qx.clone()
    for i in range(B):
        mode = torch.randint(0, 4, (1,), generator=g).item()
        if mode == 0:   # Mask Top Half
            qx_masked[i, :, :H//2, :] = 0.0
        elif mode == 1: # Mask Bottom Half
            qx_masked[i, :, H//2:, :] = 0.0
        elif mode == 2: # Mask Left Half
            qx_masked[i, :, :, :W//2] = 0.0
        else:           # Mask Right Half
            qx_masked[i, :, :, W//2:] = 0.0
    return qx_masked

def apply_position_shift(qx):
    B, C, H, W = qx.shape
    shift_h = int(round(0.25 * H))
    shift_w = int(round(0.25 * W))
    qx_shifted = torch.zeros_like(qx)
    qx_shifted[:, :, shift_h:, shift_w:] = qx[:, :, :H-shift_h, :W-shift_w]
    return qx_shifted

def eval_model_all_conditions(model, ds_test, k_shot=5, seed=42):
    model.eval()
    acc_std, acc_mask, acc_shift = [], [], []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=5, k_shot=k_shot, q_query=15, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            protos = model.compute_prototypes(sx, sy, n=5)

            # 1. Standard Condition
            preds_std = model.predict_proto(qx, protos)
            acc_std.append((preds_std.argmax(1) == qy).float().mean().item() * 100.0)

            # 2. Partial-Object 50% Mask Condition
            qx_mask = apply_partial_mask(qx, seed=ep_seed)
            preds_mask = model.predict_proto(qx_mask, protos)
            acc_mask.append((preds_mask.argmax(1) == qy).float().mean().item() * 100.0)

            # 3. Position-Shift 25% Translation Condition
            qx_shift = apply_position_shift(qx)
            preds_shift = model.predict_proto(qx_shift, protos)
            acc_shift.append((preds_shift.argmax(1) == qy).float().mean().item() * 100.0)

    return np.mean(acc_std), np.mean(acc_mask), np.mean(acc_shift)

def meta_train(model_fn, ds_train, k_shot=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_fn().to(device)
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

def compute_pairwise_distances(centers):
    B, K, _ = centers.shape
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(torch.norm(centers[:, i] - centers[:, j], dim=-1))
    return torch.stack(dists, dim=-1).mean().item()

def get_diagnostics(model, ds_test):
    model.eval()
    base_c = get_exp_p1_base_centers().to(device)
    pair_dists, base_drift, all_rel_tokens = [], [], []
    with torch.no_grad():
        for ep in range(50):
            sx, _, _, _, _ = ds_test.sample_episode(n_way=5, k_shot=5, q_query=15, seed=5000+ep)
            sx = sx.to(device)
            _, centers, rel_toks = model.extract_with_rel_tokens(sx)
            drift   = torch.norm(centers - base_c.unsqueeze(0), dim=-1).mean().item()
            p_dist  = compute_pairwise_distances(centers)

            pair_dists.append(p_dist)
            base_drift.append(drift)
            all_rel_tokens.append(rel_toks.cpu())

    all_rel_tokens = torch.cat(all_rel_tokens, dim=0)
    rel_var = torch.var(all_rel_tokens, dim=0).mean().item()
    return np.mean(pair_dists), np.mean(base_drift), rel_var

# =====================================================================
# MAIN BENCHMARK EXECUTION
# =====================================================================
def run_benchmark():
    print("=" * 100)
    print("EXP-F3 BENCHMARK: Windowed Adaptive Locator (window_frac = 0.50)")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 100, flush=True)

    m_f3 = IRFEExpF3_CIFAR(window_frac=0.50)
    f3_params = sum(p.numel() for p in m_f3.parameters() if p.requires_grad)

    print(f"EXP-F3 Params: {f3_params:,} parameters (Identical to EXP-F1: 22,249 params)")
    print("=" * 100, flush=True)

    ds_train_cifar = RealCIFARFS(split='train', seed=42)
    ds_test_cifar  = RealCIFARFS(split='test',  seed=42)
    ds_train_mini  = RealMiniImageNet(split='train', seed=42)
    ds_test_mini   = RealMiniImageNet(split='test',  seed=42)

    res_data = {
        "CIFAR": {1: {}, 5: {}},
        "MINI":  {1: {}, 5: {}}
    }
    diag_data = {}

    models_to_test = [
        ("F3_Windowed_0.50", lambda: IRFEExpF3_CIFAR(window_frac=0.50), lambda: IRFEExpF3_Native(window_frac=0.50))
    ]

    for model_name, cifar_cls, mini_cls in models_to_test:
        print(f"\n==========================================================================================")
        print(f"EXECUTING MODEL: {model_name}")
        print(f"==========================================================================================")

        for ds_name, ds_train, ds_test, cls_fn, key_ds in [
            ("CIFAR-FS (32x32)", ds_train_cifar, ds_test_cifar, cifar_cls, "CIFAR"),
            ("MiniImageNet Native (84x84)", ds_train_mini, ds_test_mini, mini_cls, "MINI")
        ]:
            for shot in [1, 5]:
                print(f"\n--- Running {ds_name} | {shot}-Shot | {model_name} ---", flush=True)
                cond_std, cond_mask, cond_shift = [], [], []

                for seed in FIVE_SEEDS:
                    m = meta_train(cls_fn, ds_train, k_shot=shot, seed=seed)
                    a_std, a_mask, a_shift = eval_model_all_conditions(m, ds_test, k_shot=shot, seed=seed)
                    cond_std.append(a_std)
                    cond_mask.append(a_mask)
                    cond_shift.append(a_shift)

                    if seed == 1 and shot == 5:
                        p_dist, drift, r_var = get_diagnostics(m, ds_test)
                        diag_data[f"{key_ds}_{model_name}"] = (p_dist, drift, r_var)

                    print(f"  [{ds_name} {shot}S Seed {seed:3d}] Std: {a_std:.2f}% | Mask50%: {a_mask:.2f}% | Shift25%: {a_shift:.2f}%", flush=True)

                res_data[key_ds][shot][model_name] = {
                    "std": cond_std,
                    "mask": cond_mask,
                    "shift": cond_shift
                }

    # REUSED EXP-F1 AND BASELINE NUMBERS FOR COMPARISON
    reported_exp_f1 = {
        "CIFAR": {
            1: {"std": [37.40, 36.93, 38.12, 36.27, 37.68], "mask": [31.82, 30.93, 31.52, 30.81, 31.58], "shift": [32.12, 31.25, 32.48, 31.12, 32.10]},
            5: {"std": [52.93, 51.52, 54.11, 50.91, 51.87], "mask": [46.12, 44.59, 46.73, 45.39, 45.98], "shift": [43.83, 42.35, 44.51, 44.31, 43.35]}
        },
        "MINI": {
            1: {"std": [32.60, 33.31, 35.62, 32.85, 34.32], "mask": [29.35, 29.62, 30.74, 29.54, 29.73], "shift": [30.38, 31.02, 32.88, 31.00, 32.58]},
            5: {"std": [48.96, 50.47, 51.38, 49.90, 49.98], "mask": [43.49, 43.57, 45.20, 44.13, 44.60], "shift": [45.17, 44.81, 46.59, 45.67, 46.33]}
        }
    }

    reported_baseline = {
        "CIFAR": {
            1: {"std": [41.31, 41.20, 42.49, 37.91, 40.82], "mask": [30.81, 28.46, 31.18, 29.44, 30.12], "shift": [29.81, 29.74, 32.52, 31.65, 28.42]},
            5: {"std": [57.40, 56.93, 60.75, 57.27, 57.36], "mask": [39.76, 39.05, 41.15, 42.21, 36.98], "shift": [40.97, 39.13, 40.75, 42.68, 37.66]}
        },
        "MINI": {
            1: {"std": [36.28, 37.50, 39.30, 34.40, 34.65], "mask": [29.61, 29.55, 30.30, 26.97, 27.51], "shift": [31.94, 32.76, 33.30, 30.23, 31.31]},
            5: {"std": [53.82, 53.57, 55.01, 53.54, 53.84], "mask": [37.92, 38.61, 39.62, 38.89, 40.88], "shift": [43.46, 42.28, 42.79, 43.04, 41.65]}
        }
    }

    # PRINT COMPARATIVE MASTER TABLES
    print("\n\n" + "=" * 100)
    print("EXP-F3 MASTER REPORT TABLES (5-SEED COMPARISON)")
    print("=" * 100)

    for key_ds, ds_label in [("CIFAR", "CIFAR-FS (32x32)"), ("MINI", "MiniImageNet Native (84x84)")]:
        for shot in [1, 5]:
            print(f"\n==========================================================================================")
            print(f"MASTER TABLE: {ds_label} | {shot}-SHOT")
            print(f"==========================================================================================")
            
            for cond_key, cond_label in [
                ("std", "Standard Unmodified Query"),
                ("mask", "Partial-Object Robustness (50% Masked Query)"),
                ("shift", "Position-Shift Robustness (25% Shifted Query)")
            ]:
                print(f"\n--- Condition: {cond_label} ---")
                print(f"{'Seed':<6} | {'Exp-P1 Base (~22.2k)':<20} | {'EXP-F1 (w=1.0) (~22.2k)':<22} | {'EXP-F3 (w=0.50) (~22.2k)':<24}")
                print("-" * 80)
                p1_v = reported_baseline[key_ds][shot][cond_key]
                f1_v = reported_exp_f1[key_ds][shot][cond_key]
                f3_v = res_data[key_ds][shot]["F3_Windowed_0.50"][cond_key]

                for idx, seed in enumerate(FIVE_SEEDS):
                    print(f"{seed:<6} | {p1_v[idx]:18.2f}% | {f1_v[idx]:20.2f}% | {f3_v[idx]:22.2f}%")
                print("-" * 80)
                
                # 5-Seed Mean ± Std
                p1_m5, p1_s5 = np.mean(p1_v), np.std(p1_v)
                f1_m5, f1_s5 = np.mean(f1_v), np.std(f1_v)
                f3_m5, f3_s5 = np.mean(f3_v), np.std(f3_v)
                print(f"{'5-SEED':<6} | {p1_m5:4.2f}% ± {p1_s5:4.2f}%     | {f1_m5:4.2f}% ± {f1_s5:4.2f}%       | {f3_m5:4.2f}% ± {f3_s5:4.2f}%")

                # 4-Seed Mean ± Std (No Seed 21)
                idx_no21 = [i for i, s in enumerate(FIVE_SEEDS) if s != 21]
                p1_no21, f1_no21, f3_no21 = [p1_v[i] for i in idx_no21], [f1_v[i] for i in idx_no21], [f3_v[i] for i in idx_no21]
                print(f"{'NO-21':<6} | {np.mean(p1_no21):4.2f}% ± {np.std(p1_no21):4.2f}%     | {np.mean(f1_no21):4.2f}% ± {np.std(f1_no21):4.2f}%       | {np.mean(f3_no21):4.2f}% ± {np.std(f3_no21):4.2f}%")

    print("\n\n" + "=" * 100)
    print("MECHANISTIC & SPATIAL SLOT DYNAMICS SUMMARY (5-SHOT, SEED 1)")
    print("=" * 100)
    print(f"{'Dataset & Model':<30} | {'Pairwise Slot Dist':<20} | {'Base Center Drift':<20} | {'Relation Token Var':<20}")
    print("-" * 96)
    for key in sorted(diag_data.keys()):
        p_dist, drift, r_var = diag_data[key]
        print(f"{key:<30} | {p_dist:18.4f} | {drift:18.4f} | {r_var:18.6f}")
    print("=" * 100)

if __name__ == '__main__':
    run_benchmark()

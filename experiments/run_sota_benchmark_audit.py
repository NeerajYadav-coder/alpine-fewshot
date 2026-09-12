# run_sota_benchmark_audit.py
# Benchmark IRFE v6.0 against CNN, ProtoNet, and Vision Transformer (ViT)
# Under strict parameter parity (<35k parameters) on a Difficult Few-Shot Dataset.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from irfe_v50 import IRFEv50
from irfe_v60 import IRFEv60_FourierInvariant
from run_level4_data_efficiency_gate import RealisticComplexFewShotDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing SOTA & ViT Comparison Benchmark on device: {device}")

# ---------------------------------------------------------
# 1. Parameter-Matched Standard 4-Layer ConvNet (ProtoNet)
# ---------------------------------------------------------
class ConvNet4_ProtoNet(nn.Module):
    """Standard 4-Layer Convolutional Network (ProtoNet Backbone) - Matched to ~29k params"""
    def __init__(self, in_channels=3, embed_dim=16):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU())
        self.fc = nn.Linear(embed_dim * 3 * 3, 128)

    def extract_patch_features(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.fc(h.view(x.size(0), -1))

# ---------------------------------------------------------
# 2. Parameter-Matched Vision Transformer (ViT)
# ---------------------------------------------------------
class CompactVisionTransformer(nn.Module):
    """
    Compact Vision Transformer (ViT) matched to ~29k parameter budget.
    Patch size 4x4 -> 49 patches for 28x28 image.
    """
    def __init__(self, in_channels=3, patch_size=4, embed_dim=32, num_heads=2, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 256)

    def extract_patch_features(self, x):
        B, C, H, W = x.shape
        # Flatten into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        
        tokens = self.patch_embed(patches) # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) # [B, N+1, D]
        tokens = tokens + self.pos_embed
        
        out = self.transformer(tokens)
        cls_rep = out[:, 0]
        return self.fc(cls_rep)

# ---------------------------------------------------------
# 3. Model Setup & Parameter Count Audit
# ---------------------------------------------------------
models = {
    '1. ConvNet-4 (ProtoNet Baseline)': ConvNet4_ProtoNet,
    '2. Vision Transformer (ViT)': CompactVisionTransformer,
    '3. IRFE v5.0 Baseline (Spatial)': IRFEv50,
    '4. IRFE v6.0 Fourier Invariant (Champion)': IRFEv60_FourierInvariant
}

print("\n--- PARAMETER AUDIT (PARITY VERIFICATION) ---")
for m_name, m_cls in models.items():
    m = m_cls()
    p_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Model: {m_name:42s} | Parameters: {p_count}")

# ---------------------------------------------------------
# 4. Multi-Seed Benchmark Protocol (High-Difficulty Dataset)
# ---------------------------------------------------------
training_seeds = [101, 202, 303, 404, 505]
ds_test_difficult = RealisticComplexFewShotDataset(n_classes=40, images_per_class=100, seed=8888)

shift_steps = [0, 1, 2, 4, 6, 8]
audit_results = {m: {s: [] for s in shift_steps} for m in models}

print("\n======================================================================")
print("EXECUTING SOTA & ViT BENCHMARK ON DIFFICULT FEW-SHOT DATASET")
print("======================================================================\n")

for m_name, m_cls in models.items():
    print(f"--- Auditing Model: {m_name} ---")
    
    for seed in training_seeds:
        ds_train_sub = RealisticComplexFewShotDataset(n_classes=64, images_per_class=600, data_fraction=0.01, seed=seed)
        model = m_cls().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Meta-train (100 episodes)
        model.train()
        for ep in range(100):
            supp_x, supp_y, qry_x, qry_y = ds_train_sub.sample_episode(n_way=5, k_shot=1, q_query=15, seed=seed*100+ep)
            supp_x, supp_y = supp_x.to(device), supp_y.to(device)
            qry_x, qry_y = qry_x.to(device), qry_y.to(device)
            
            opt.zero_grad()
            if hasattr(model, 'extract_patch_features'):
                supp_f = model.extract_patch_features(supp_x)
                qry_f  = model.extract_patch_features(qry_x)
            else:
                supp_f = model.extract_features(supp_x)
                qry_f  = model.extract_features(qry_x)
                
            protos = torch.stack([supp_f[supp_y == c].mean(0) for c in range(5)])
            dists = torch.cdist(qry_f, protos)
            loss = F.cross_entropy(-dists, qry_y)
            loss.backward()
            opt.step()
            
        # Audit on Difficult Test Set across shift steps
        model.eval()
        with torch.no_grad():
            for s_val in shift_steps:
                accs = []
                for ep in range(50):
                    supp_x, supp_y, qry_x, qry_y = ds_test_difficult.sample_episode(n_way=5, k_shot=1, q_query=15, seed=7777+ep)
                    supp_x, supp_y = supp_x.to(device), supp_y.to(device)
                    qry_x, qry_y = qry_x.to(device), qry_y.to(device)
                    
                    qry_x_shifted = torch.roll(qry_x, shifts=(s_val, s_val), dims=(2, 3)) if s_val > 0 else qry_x
                    if hasattr(model, 'extract_patch_features'):
                        supp_f = model.extract_patch_features(supp_x)
                        qry_f  = model.extract_patch_features(qry_x_shifted)
                    else:
                        supp_f = model.extract_features(supp_x)
                        qry_f  = model.extract_features(qry_x_shifted)
                        
                    protos = torch.stack([supp_f[supp_y == c].mean(0) for c in range(5)])
                    preds = (-torch.cdist(qry_f, protos)).argmax(dim=1)
                    accs.append((preds == qry_y).float().mean().item())
                audit_results[m_name][s_val].append(np.mean(accs))
                
        s0 = audit_results[m_name][0][-1] * 100.0
        s1 = audit_results[m_name][1][-1] * 100.0
        s2 = audit_results[m_name][2][-1] * 100.0
        s4 = audit_results[m_name][4][-1] * 100.0
        s6 = audit_results[m_name][6][-1] * 100.0
        s8 = audit_results[m_name][8][-1] * 100.0
        print(f"    Seed {seed} | 0px: {s0:5.2f}% | 1px: {s1:5.2f}% | 2px: {s2:5.2f}% | 4px: {s4:5.2f}% | 6px: {s6:5.2f}% | 8px: {s8:5.2f}%")

# Compute Statistics
summary_stats = {m: {} for m in models}
for m_name in models:
    for s_val in shift_steps:
        m_val = np.mean(audit_results[m_name][s_val]) * 100.0
        std_val = np.std(audit_results[m_name][s_val]) * 100.0
        summary_stats[m_name][s_val] = (m_val, std_val)

# Write Markdown Report
report_md = f"""# SOTA & Vision Transformer Comparison Audit Report

## 1. Experimental Protocol
* **Objective:** Benchmark IRFE v6.0 against standard CNN (ConvNet-4 / ProtoNet) and Vision Transformer (ViT) under strict parameter parity (~28k-32k parameters).
* **Evaluated Models:** ConvNet-4 (ProtoNet), Vision Transformer (ViT), IRFE v5.0, IRFE v6.0 Fourier Invariant.
* **Dataset:** High-Difficulty Complex Few-Shot Dataset (40 Classes).
* **Shift Steps:** 0px, 1px, 2px, 4px, 6px, 8px.
* **Seeds:** 5 Independent Seeds (101, 202, 303, 404, 505).

---

## 2. Spatial Robustness Curve Matrix (Mean ± SD across 5 Seeds)

| Model Architecture | Parameters | 0px (Clean) | 1px Shift | 2px Shift | 4px Shift | 6px Shift | 8px Shift | AUC (Rob) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

for m_name in models:
    p_cnt = sum(p.numel() for p in models[m_name]().parameters() if p.requires_grad)
    vals = [summary_stats[m_name][s][0] for s in shift_steps]
    auc_val = np.trapezoid(vals, x=shift_steps) / 8.0 # Normalized AUC
    report_md += f"| **{m_name}** | `{p_cnt:,}` | `{summary_stats[m_name][0][0]:.2f}% ± {summary_stats[m_name][0][1]:.2f}%` | `{summary_stats[m_name][1][0]:.2f}% ± {summary_stats[m_name][1][1]:.2f}%` | `{summary_stats[m_name][2][0]:.2f}% ± {summary_stats[m_name][2][1]:.2f}%` | `{summary_stats[m_name][4][0]:.2f}% ± {summary_stats[m_name][4][1]:.2f}%` | `{summary_stats[m_name][6][0]:.2f}% ± {summary_stats[m_name][6][1]:.2f}%` | `{summary_stats[m_name][8][0]:.2f}% ± {summary_stats[m_name][8][1]:.2f}%` | **`{auc_val:.2f}%`** |\n"

with open("scientific_lock_sota_vit_comparison_report.md", "w") as f:
    f.write(report_md)

print("\n======================================================================")
print("AUDIT COMPLETE. REPORT SAVED TO scientific_lock_sota_vit_comparison_report.md")
print("======================================================================\n")

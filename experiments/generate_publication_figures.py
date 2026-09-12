# generate_publication_figures.py
#
# PUBLICATION-READY FIGURE GENERATOR FOR IRFE
# Generates high-resolution PNG figures (300 DPI) for:
# - Figure 1: Patch Localization Diagnostic (Baseline vs EXP-F3 on Standard/Masked/Shifted images)
# - Figure 2: Capacity vs Accuracy Curve (Log-scale params, sweet spot, error bars)
# - Figure 3: Learning & Sample-Efficiency Curves (Episodes 0-250, shaded 3-seed variance)
# - Figure 4: Gabor Energy Map & Locator Mechanics (Original, Energy Map, Adaptive Centers)

import json
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_fs_dataset import RealCIFARFS
from mini_imagenet_mmap_loader import MiniImageNetMmap as RealMiniImageNet
from run_exp_f3_benchmark import IRFEExpF3_CIFAR, IRFEExpF3_Native, get_exp_p1_base_centers, WideWindowAdaptivePatchLocator

# Matplotlib styling for publication quality
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "."

# Helper to denormalize RGB tensors for visualization
def tensor_to_img(t):
    img = t.cpu().detach().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

# Apply 50% partial mask (Condition 2)
def apply_mask(img):
    c = img.clone()
    H, W = c.shape[1], c.shape[2]
    c[:, :, :W//2] = 0.0 # 50% vertical half mask
    return c

# Apply 25% position shift (Condition 3)
def apply_shift(img):
    c = torch.zeros_like(img)
    H, W = img.shape[1], img.shape[2]
    shift_h, shift_w = H // 4, W // 4
    c[:, shift_h:, shift_w:] = img[:, :-shift_h, :-shift_w]
    return c

# =====================================================================
# FIGURE 1: Patch Localization Diagnostic
# =====================================================================
def generate_figure1():
    print("Generating Figure 1: Patch Localization Diagnostic...", flush=True)
    m_cifar = IRFEExpF3_CIFAR().to(device)
    ckpt_path = "checkpoints/exp_f3_cifar_1shot_seed1.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m_cifar.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m_cifar.eval()

    ds = RealCIFARFS(split="test", seed=1)
    sx, sy, qx, qy, _ = ds.sample_episode(n_way=5, k_shot=1, q_query=15, seed=123)

    imgs = [
        ("Standard Sample 1", qx[0]),
        ("Standard Sample 2", qx[1]),
        ("50% Masked Sample 1", apply_mask(qx[0])),
        ("50% Masked Sample 2", apply_mask(qx[1])),
        ("25% Shifted Sample 1", apply_shift(qx[0])),
        ("25% Shifted Sample 2", apply_shift(qx[1])),
    ]

    base_centers = get_exp_p1_base_centers() # (5, 2)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()

    colors_exp = ['cyan', 'lime', 'yellow', 'magenta', 'orange']

    for i, (title, img_t) in enumerate(imgs):
        ax = axes[i]
        img_np = tensor_to_img(img_t)
        ax.imshow(img_np)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')

        H, W = img_t.shape[1], img_t.shape[2]

        # Compute EXP-F3 adaptive centers
        with torch.no_grad():
            x_in = img_t.unsqueeze(0).to(device)
            gabor_out = m_cifar.encoder.gabor(x_in)
            energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)
            adapt_centers = m_cifar.locator(energy_map).squeeze(0).cpu() # (5, 2)

        # Plot Baseline Fixed Centers (Red)
        for k in range(5):
            cy_b = (base_centers[k, 0].item() + 1.0) * 0.5 * (H - 1)
            cx_b = (base_centers[k, 1].item() + 1.0) * 0.5 * (W - 1)
            ax.plot(cx_b, cy_b, 'rx', markersize=9, markeredgewidth=2)
            # 16x16 patch box for baseline
            rect_b = patches.Rectangle((cx_b - 8, cy_b - 8), 16, 16, linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect_b)

        # Plot EXP-F3 Adaptive Centers (Colored)
        for k in range(5):
            cy_a = (adapt_centers[k, 1].item() + 1.0) * 0.5 * (H - 1)
            cx_a = (adapt_centers[k, 0].item() + 1.0) * 0.5 * (W - 1)
            ax.plot(cx_a, cy_a, 'o', color=colors_exp[k], markersize=7)
            rect_a = patches.Rectangle((cx_a - 8, cy_a - 8), 16, 16, linewidth=1.8, edgecolor=colors_exp[k], facecolor='none')
            ax.add_patch(rect_a)

    # Global Legend identifying fixed baseline + individual 5 patch slots
    slot_labels = ['Patch Slot 1', 'Patch Slot 2', 'Patch Slot 3', 'Patch Slot 4', 'Patch Slot 5']
    custom_lines = [
        plt.Line2D([0], [0], color='red', linestyle='--', marker='x', markersize=8, markeredgewidth=2, label='Baseline Fixed Grid')
    ] + [
        plt.Line2D([0], [0], color=colors_exp[k], linestyle='-', marker='o', markersize=7, label=f'EXP-F3 {slot_labels[k]}')
        for k in range(5)
    ]
    fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6, frameon=True, fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(OUTPUT_DIR, "figure1_patch_localization_diagnostic.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved to {fig_path}")

# =====================================================================
# FIGURE 2: Capacity vs Accuracy Curve
# =====================================================================
def generate_figure2():
    print("Generating Figure 2: Capacity vs Accuracy Curve...", flush=True)

    # Data structure: (params, cifar_mean, cifar_std, mini_mean, mini_std, is_verified, label)
    # EXP-F3 Capacity Scaling Points ONLY (ProtoNet removed from line to prevent misleading connection)
    data = [
        (22249,  40.36, 1.26, 35.88, 0.88, True,  "EXP-F3 (22k)"),
        (29962,  40.20, 0.00, 35.50, 0.00, False, "EXP-F3-30k"),
        (34917,  40.61, 1.34, 36.76, 0.40, True,  "EXP-F3-35k (Sweet Spot)"),
        (48887,  41.06, 0.00, 36.10, 0.00, False, "EXP-F3-47k"),
        (59883,  41.73, 0.00, 35.95, 0.00, False, "EXP-F3-60k"),
        (122897, 40.79, 0.66, 35.95, 0.84, True,  "EXP-F3-Large (123k)"),
        (148771, 40.40, 0.00, 36.61, 0.00, False, "EXP-F3-149k"),
        (196035, 40.16, 0.00, 36.50, 0.00, False, "EXP-F3-196k"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    params_cifar = [d[0] for d in data]
    cifar_acc    = [d[1] for d in data]

    params_mini  = [d[0] for d in data]
    mini_acc     = [d[3] for d in data]

    # EXP-F3 CIFAR-FS Series
    ax.plot(params_cifar, cifar_acc, 'o--', color='#1f77b4', label='EXP-F3 Scaling (CIFAR-FS)', alpha=0.75, linewidth=1.8)
    for d in data:
        p, acc, err, _, _, ver, _ = d
        if ver:
            ax.errorbar(p, acc, yerr=err, fmt='o', color='#1f77b4', capsize=4, capthick=1.5, elinewidth=1.5)
        else:
            ax.plot(p, acc, 'o', markerfacecolor='white', markeredgecolor='#1f77b4', markersize=6)

    # EXP-F3 MiniImageNet Series
    ax.plot(params_mini, mini_acc, 's-', color='#ff7f0e', label='EXP-F3 Scaling (MiniImageNet)', alpha=0.75, linewidth=1.8)
    for d in data:
        p, _, _, acc, err, ver, _ = d
        if ver:
            ax.errorbar(p, acc, yerr=err, fmt='s', color='#ff7f0e', capsize=4, capthick=1.5, elinewidth=1.5)
        else:
            ax.plot(p, acc, 's', markerfacecolor='white', markeredgecolor='#ff7f0e', markersize=6)

    # Horizontal Reference Lines for ProtoNet Baseline (47,630 params)
    ax.axhline(y=40.45, color='#1f77b4', linestyle=':', linewidth=1.8, alpha=0.85)
    ax.text(20500, 40.52, 'ProtoNet CIFAR-FS Baseline (40.45%, 47.6k params)', color='#1f77b4', fontsize=9, fontweight='bold')

    ax.axhline(y=35.25, color='#ff7f0e', linestyle=':', linewidth=1.8, alpha=0.85)
    ax.text(20500, 34.80, 'ProtoNet MiniImageNet Baseline (35.25%, 47.6k params)', color='#d95f02', fontsize=9, fontweight='bold')

    # Annotate Sweet Spot (34,917 params)
    ax.annotate('EXP-F3-35k Sweet Spot\n(34,917 params: 36.76% Mini, 40.61% CIFAR)',
                xy=(34917, 36.76), xytext=(45000, 38.2),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.9))

    ax.set_xscale('log')
    ax.set_xlabel('Trainable Parameters (Log Scale)', fontweight='bold')
    ax.set_ylabel('1-Shot Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Capacity vs. Accuracy Curve across Model Scale Points', fontweight='bold', pad=12)

    ax.set_xticks([20000, 35000, 50000, 100000, 200000])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}k".format(int(x//1000))))
    ax.set_ylim(33.0, 43.5)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', linestyle='--', marker='o', label='EXP-F3 Scaling (CIFAR-FS 1-Shot)'),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle='-', marker='s', label='EXP-F3 Scaling (MiniImageNet Native 1-Shot)'),
        plt.Line2D([0], [0], color='#1f77b4', linestyle=':', linewidth=1.8, label='ProtoNet CIFAR-FS Baseline (47,630 params)'),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle=':', linewidth=1.8, label='ProtoNet MiniImageNet Baseline (47,630 params)'),
        plt.Line2D([0], [0], color='black', linestyle='none', marker='o', label='Solid Marker: 5-Seed Verified (±std error bars)'),
        plt.Line2D([0], [0], color='black', linestyle='none', marker='o', markerfacecolor='white', label='Hollow Marker: Single-Seed Cheap Probe')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9.0)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figure2_capacity_vs_accuracy.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {fig_path}")

# =====================================================================
# FIGURE 3: Learning/Sample-Efficiency Curves
# =====================================================================
def generate_figure3():
    print("Generating Figure 3: Learning & Sample-Efficiency Curves...", flush=True)

    manifest_path = "checkpoints/sota_learning_curve_seeds_manifest.json"
    if not os.path.exists(manifest_path):
        print("Warning: Learning curve manifest not found, skipping Fig 3.")
        return

    with open(manifest_path) as f:
        data = json.load(f)

    episodes = [0, 50, 100, 150, 200, 250]
    models   = ["EXP-F3", "ProtoNet", "MAML", "RelationNet"]
    colors   = {
        "EXP-F3": "#d62728",      # Crimson red
        "ProtoNet": "#1f77b4",    # Blue
        "MAML": "#9467bd",        # Purple
        "RelationNet": "#2ca02c"  # Green
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (ds_key, ds_title) in enumerate([("cifar", "CIFAR-FS (5-Shot)"), ("mini", "MiniImageNet Native (5-Shot)")]):
        ax = axes[idx]
        ax.set_title(f"Learning-Curve Trajectory: {ds_title}", fontweight='bold')
        ax.set_xlabel("Meta-Training Episodes", fontweight='bold')
        ax.set_ylabel("Evaluation Accuracy (%)", fontweight='bold')

        for m in models:
            m_data = data["curves"][ds_key][m]
            # m_data structure: seed -> ep -> acc
            ep_means = []
            ep_stds  = []
            for ep in episodes:
                accs = [m_data[str(s)][str(ep)] for s in [1, 7, 42]]
                ep_means.append(np.mean(accs))
                ep_stds.append(np.std(accs))

            ep_means = np.array(ep_means)
            ep_stds  = np.array(ep_stds)

            ax.plot(episodes, ep_means, 'o-', label=m, color=colors[m], linewidth=2.0, markersize=5)
            ax.fill_between(episodes, ep_means - ep_stds, ep_means + ep_stds, color=colors[m], alpha=0.15)

        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc='lower right' if ds_key == "cifar" else 'lower right', frameon=True, fontsize=10)
        ax.set_xticks(episodes)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figure3_learning_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved to {fig_path}")

# =====================================================================
# FIGURE 4: Gabor Energy Map / Locator Visualization
# =====================================================================
def generate_figure4():
    print("Generating Figure 4: Gabor Energy Map & Locator Mechanics...", flush=True)

    m_cifar = IRFEExpF3_CIFAR().to(device)
    ckpt_path = "checkpoints/exp_f3_cifar_1shot_seed1.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m_cifar.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    m_cifar.eval()

    ds = RealCIFARFS(split="test", seed=42)
    sx, sy, qx, qy, _ = ds.sample_episode(n_way=5, k_shot=1, q_query=15, seed=777)

    sample_indices = [0, 3, 7] # 3 sample images
    colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange']

    fig, axes = plt.subplots(3, 3, figsize=(10, 9.5))

    for idx, s_idx in enumerate(sample_indices):
        img_t = qx[s_idx]
        img_np = tensor_to_img(img_t)
        H, W = img_t.shape[1], img_t.shape[2]

        with torch.no_grad():
            x_in = img_t.unsqueeze(0).to(device)
            gabor_out = m_cifar.encoder.gabor(x_in)
            energy_map = gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True).squeeze().cpu().numpy()
            adapt_centers = m_cifar.locator(gabor_out[:, 3:7].abs().sum(dim=1, keepdim=True)).squeeze(0).cpu()

        # (a) Original Image
        ax1 = axes[idx, 0]
        ax1.imshow(img_np)
        ax1.set_title(f"Sample {idx+1}: Input RGB", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # (b) Gabor Energy Map
        ax2 = axes[idx, 1]
        im_e = ax2.imshow(energy_map, cmap='inferno')
        ax2.set_title(f"Sample {idx+1}: Gabor Energy E(y,x)", fontsize=11, fontweight='bold')
        ax2.axis('off')
        fig.colorbar(im_e, ax=ax2, fraction=0.046, pad=0.04)

        # (c) Overlay Adaptive Centers & Patches
        ax3 = axes[idx, 2]
        ax3.imshow(img_np)
        ax3.set_title(f"Sample {idx+1}: Energy-Guided Patches", fontsize=11, fontweight='bold')
        ax3.axis('off')

        for k in range(5):
            cy = (adapt_centers[k, 1].item() + 1.0) * 0.5 * (H - 1)
            cx = (adapt_centers[k, 0].item() + 1.0) * 0.5 * (W - 1)
            ax3.plot(cx, cy, 'o', color=colors[k], markersize=7)
            rect = patches.Rectangle((cx - 8, cy - 8), 16, 16, linewidth=1.8, edgecolor=colors[k], facecolor='none')
            ax3.add_patch(rect)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figure4_gabor_energy_locator.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {fig_path}")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    print("\nALL 4 PUBLICATION FIGURES GENERATED SUCCESSFULLY.")

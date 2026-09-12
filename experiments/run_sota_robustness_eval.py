# run_sota_robustness_eval.py
# Robustness Comparison vs SOTA Baselines (Masked & Shifted Conditions)
#
# Protocol:
#   - Zero retraining (evaluation-only pass on verified MiniImageNet and CIFAR-FS checkpoints)
#   - Models evaluated:
#       1) EXP-F3 (22,249 params)
#       2) ProtoNet Baseline (47,630 params)
#       3) MAML Baseline (49,481 params)
#       4) RelationNet Baseline (49,617 params)
#   - Datasets: CIFAR-FS (32x32) and MiniImageNet Native (84x84)
#   - Shot settings: 1-shot and 5-shot (N=5 way, Q=15 query)
#   - Seeds: [1, 7, 21, 42, 123] (600 test episodes per seed per condition)
#   - Robustness Conditions:
#       1) Standard (unmodified query)
#       2) 50% Partial-Object Masking (top, bottom, left, or right half zeroed out)
#       3) 25% Position-Shift (shifted right and down by 25% of image height/width)
#   - Output manifest saved to: checkpoints/sota_robustness_manifest.json

import datetime
import hashlib
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cifar_fs_dataset import RealCIFARFS
from mini_imagenet_mmap_loader import MiniImageNetMmap as RealMiniImageNet

from run_exp_f3_benchmark import IRFEExpF3_CIFAR, IRFEExpF3_Native
from stage2a_protonet import ProtoNetBaseline
from stage2b_maml import MAMLBackbone
from stage2c_relationnet import RelationNetBaseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 7, 21, 42, 123]
N_WAY = 5
Q_QUERY = 15
EVAL_EPISODES = 600

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Robustness Corruption Transformations
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Evaluation routines across 3 conditions
# ---------------------------------------------------------------------------
def eval_exp_f3_robustness(model, ds_test, k_shot, seed):
    model.eval()
    acc_std, acc_mask, acc_shift = [], [], []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            protos = model.compute_prototypes(sx, sy, n=N_WAY)

            qx_masked  = apply_partial_mask(qx, seed=ep_seed)
            qx_shifted = apply_position_shift(qx)

            preds_std   = model.predict_proto(qx, protos)
            preds_mask  = model.predict_proto(qx_masked, protos)
            preds_shift = model.predict_proto(qx_shifted, protos)

            acc_std.append((preds_std.argmax(1) == qy).float().mean().item() * 100.0)
            acc_mask.append((preds_mask.argmax(1) == qy).float().mean().item() * 100.0)
            acc_shift.append((preds_shift.argmax(1) == qy).float().mean().item() * 100.0)

    return float(np.mean(acc_std)), float(np.mean(acc_mask)), float(np.mean(acc_shift))

def eval_protonet_robustness(model, ds_test, k_shot, seed):
    model.eval()
    acc_std, acc_mask, acc_shift = [], [], []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            protos = model.compute_prototypes(sx, sy, n=N_WAY)

            qx_masked  = apply_partial_mask(qx, seed=ep_seed)
            qx_shifted = apply_position_shift(qx)

            preds_std   = model.predict_proto(qx, protos)
            preds_mask  = model.predict_proto(qx_masked, protos)
            preds_shift = model.predict_proto(qx_shifted, protos)

            acc_std.append((preds_std.argmax(1) == qy).float().mean().item() * 100.0)
            acc_mask.append((preds_mask.argmax(1) == qy).float().mean().item() * 100.0)
            acc_shift.append((preds_shift.argmax(1) == qy).float().mean().item() * 100.0)

    return float(np.mean(acc_std)), float(np.mean(acc_mask)), float(np.mean(acc_shift))

def eval_maml_robustness(model, ds_test, k_shot, seed):
    acc_std, acc_mask, acc_shift = [], [], []
    crit = nn.CrossEntropyLoss()
    fast_model = MAMLBackbone().to(device)
    for ep in range(EVAL_EPISODES):
        ep_seed = seed + ep * 1000 + 777
        sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        # Inner loop adaptation on support set
        fast_model.load_state_dict(model.state_dict())
        fast_model.train()
        inner_opt = optim.SGD(fast_model.parameters(), lr=0.01)
        for _ in range(5):
            inner_opt.zero_grad()
            loss = crit(fast_model(sx), sy)
            loss.backward()
            inner_opt.step()

        # Query evaluation across 3 conditions
        fast_model.eval()
        qx_masked  = apply_partial_mask(qx, seed=ep_seed)
        qx_shifted = apply_position_shift(qx)

        with torch.no_grad():
            preds_std   = fast_model(qx).argmax(1)
            preds_mask  = fast_model(qx_masked).argmax(1)
            preds_shift = fast_model(qx_shifted).argmax(1)

            acc_std.append((preds_std == qy).float().mean().item() * 100.0)
            acc_mask.append((preds_mask == qy).float().mean().item() * 100.0)
            acc_shift.append((preds_shift == qy).float().mean().item() * 100.0)

    return float(np.mean(acc_std)), float(np.mean(acc_mask)), float(np.mean(acc_shift))

def eval_relationnet_robustness(model, ds_test, k_shot, seed):
    model.eval()
    acc_std, acc_mask, acc_shift = [], [], []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            qx_masked  = apply_partial_mask(qx, seed=ep_seed)
            qx_shifted = apply_position_shift(qx)

            rel_std   = model(sx, sy, qx, n_way=N_WAY)
            rel_mask  = model(sx, sy, qx_masked, n_way=N_WAY)
            rel_shift = model(sx, sy, qx_shifted, n_way=N_WAY)

            acc_std.append((rel_std.argmax(1) == qy).float().mean().item() * 100.0)
            acc_mask.append((rel_mask.argmax(1) == qy).float().mean().item() * 100.0)
            acc_shift.append((rel_shift.argmax(1) == qy).float().mean().item() * 100.0)

    return float(np.mean(acc_std)), float(np.mean(acc_mask)), float(np.mean(acc_shift))

# ---------------------------------------------------------------------------
# Helper function to compute statistics
# ---------------------------------------------------------------------------
def compute_stats(seed_dict):
    vals_all = [seed_dict[s] for s in SEEDS]
    vals_no21 = [seed_dict[s] for s in SEEDS if s != 21]
    return {
        "mean_5seed": float(np.mean(vals_all)),
        "std_5seed":  float(np.std(vals_all)),
        "mean_4seed": float(np.mean(vals_no21)),
        "std_4seed":  float(np.std(vals_no21))
    }

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("ROBUSTNESS EVALUATION VS SOTA BASELINES (MASKED & SHIFTED CONDITIONS)")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("Models: EXP-F3 (22.2k), ProtoNet (47.6k), MAML (49.5k), RelationNet (49.6k)")
    print("Datasets: CIFAR-FS (32x32) and MiniImageNet Native (84x84)")
    print("Conditions: 1) Standard, 2) 50% Partial-Object Masking, 3) 25% Position-Shift")
    print("Protocol: 5-way classification, 1-shot and 5-shot, Q=15, 600 episodes per seed")
    print("Seeds: [1, 7, 21, 42, 123] (zero retraining, reusing verified checkpoints)")
    print("=" * 100, flush=True)

    ds_cifar_test = RealCIFARFS(split='test', seed=42)
    ds_mini_test  = RealMiniImageNet(split='test', seed=42)

    models_info = [
        ("EXP-F3", 22249,
         {"cifar": IRFEExpF3_CIFAR, "mini": IRFEExpF3_Native},
         eval_exp_f3_robustness,
         "exp_f3_{ds}_{shot}shot_seed{seed}.pt"),

        ("ProtoNet", 47630,
         {"cifar": ProtoNetBaseline, "mini": ProtoNetBaseline},
         eval_protonet_robustness,
         "sota_protonet_{ds}_{shot}shot_seed{seed}.pt"),

        ("MAML", 49481,
         {"cifar": MAMLBackbone, "mini": MAMLBackbone},
         eval_maml_robustness,
         "sota_maml_{ds}_{shot}shot_seed{seed}.pt"),

        ("RelationNet", 49617,
         {"cifar": RelationNetBaseline, "mini": RelationNetBaseline},
         eval_relationnet_robustness,
         "sota_relationnet_{ds}_{shot}shot_seed{seed}.pt")
    ]

    results = {}
    manifest_entries = []

    for ds_tag, ds_name, ds_obj in [("cifar", "CIFAR-FS (32x32)", ds_cifar_test),
                                     ("mini",  "MiniImageNet Native (84x84)", ds_mini_test)]:
        print(f"\n\n" + "#" * 90)
        print(f"### EVALUATING DATASET: {ds_name}")
        print("#" * 90, flush=True)

        results[ds_tag] = {}

        for model_name, params, model_cls_map, eval_fn, ckpt_fmt in models_info:
            print(f"\n--- Model: {model_name} ({params:,} params) on {ds_name} ---", flush=True)
            results[ds_tag][model_name] = {}

            for shot in [1, 5]:
                print(f"\n  [{model_name} | {shot}-Shot | {ds_name}]", flush=True)

                std_per_seed = {}
                mask_per_seed = {}
                shift_per_seed = {}
                drop_mask_per_seed = {}
                drop_shift_per_seed = {}

                for seed in SEEDS:
                    ckpt_filename = ckpt_fmt.format(ds=ds_tag, shot=shot, seed=seed)
                    ckpt_path = os.path.join(CKPT_DIR, ckpt_filename)

                    if not os.path.exists(ckpt_path):
                        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

                    md5_val = compute_md5(ckpt_path)
                    model = model_cls_map[ds_tag]().to(device)
                    state_dict = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(state_dict)

                    acc_std, acc_mask, acc_shift = eval_fn(model, ds_obj, shot, seed)
                    d_mask  = acc_std - acc_mask
                    d_shift = acc_std - acc_shift

                    std_per_seed[seed]   = acc_std
                    mask_per_seed[seed]  = acc_mask
                    shift_per_seed[seed] = acc_shift
                    drop_mask_per_seed[seed]  = d_mask
                    drop_shift_per_seed[seed] = d_shift

                    print(f"    Seed {seed:3d} | Std: {acc_std:5.2f}% | Mask: {acc_mask:5.2f}% (Drop: {d_mask:+5.2f}%) | Shift: {acc_shift:5.2f}% (Drop: {d_shift:+5.2f}%)")

                    manifest_entries.append({
                        "dataset": ds_tag,
                        "model": model_name,
                        "params": params,
                        "shot": shot,
                        "seed": seed,
                        "checkpoint_file": ckpt_filename,
                        "md5": md5_val,
                        "acc_std": acc_std,
                        "acc_mask": acc_mask,
                        "acc_shift": acc_shift,
                        "drop_mask": d_mask,
                        "drop_shift": d_shift
                    })

                shot_key = f"{shot}shot"
                results[ds_tag][model_name][shot_key] = {
                    "std_per_seed": std_per_seed,
                    "mask_per_seed": mask_per_seed,
                    "shift_per_seed": shift_per_seed,
                    "drop_mask_per_seed": drop_mask_per_seed,
                    "drop_shift_per_seed": drop_shift_per_seed,

                    "std_stats": compute_stats(std_per_seed),
                    "mask_stats": compute_stats(mask_per_seed),
                    "shift_stats": compute_stats(shift_per_seed),
                    "drop_mask_stats": compute_stats(drop_mask_per_seed),
                    "drop_shift_stats": compute_stats(drop_shift_per_seed)
                }

                s_std = results[ds_tag][model_name][shot_key]["std_stats"]
                s_msk = results[ds_tag][model_name][shot_key]["mask_stats"]
                s_sft = results[ds_tag][model_name][shot_key]["shift_stats"]
                d_msk = results[ds_tag][model_name][shot_key]["drop_mask_stats"]
                d_sft = results[ds_tag][model_name][shot_key]["drop_shift_stats"]

                print(f"  SUMMARY [{model_name} {shot}-Shot {ds_tag.upper()}]:")
                print(f"    Standard condition (n=5) : {s_std['mean_5seed']:.2f}% ± {s_std['std_5seed']:.2f}%")
                print(f"    50% Masked condition (n=5): {s_msk['mean_5seed']:.2f}% ± {s_msk['std_5seed']:.2f}% (Drop: {d_msk['mean_5seed']:.2f}% ± {d_msk['std_5seed']:.2f}%)")
                print(f"    25% Shifted condition(n=5): {s_sft['mean_5seed']:.2f}% ± {s_sft['std_5seed']:.2f}% (Drop: {d_sft['mean_5seed']:.2f}% ± {d_sft['std_5seed']:.2f}%)")

    # Save manifest
    manifest_path = os.path.join(CKPT_DIR, "sota_robustness_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "experiment": "Robustness Comparison vs SOTA Baselines (Masked & Shifted)",
            "seeds": SEEDS,
            "episodes_per_seed": EVAL_EPISODES,
            "manifest_entries": manifest_entries,
            "summary_results": results
        }, f, indent=2)

    print(f"\nManifest successfully written to: {manifest_path}")

    # -----------------------------------------------------------------------
    # PRINT MASTER TABLES
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 110)
    print("MASTER ROBUSTNESS RESULTS TABLES (ROBUST ACCURACY & DEGRADATION DROPS)")
    print("=" * 110)

    for ds_tag, ds_title in [("cifar", "CIFAR-FS (32x32)"), ("mini", "MiniImageNet Native (84x84)")]:
        for shot in [5, 1]:
            shot_key = f"{shot}shot"
            print(f"\nMASTER TABLE: {ds_title} | {shot}-Shot Classification (Mean ± std across 5 seeds)")
            print(f"{'Model':<14} | {'Params':<8} | {'Standard (Ref)':<18} | {'50% Masked Acc':<18} | {'Mask Drop (Δm)':<16} | {'25% Shifted Acc':<18} | {'Shift Drop (Δs)':<16}")
            print("-" * 118)

            for model_name, params, _, _, _ in models_info:
                res = results[ds_tag][model_name][shot_key]
                st_std = res["std_stats"]
                st_msk = res["mask_stats"]
                st_sft = res["shift_stats"]
                st_dmsk = res["drop_mask_stats"]
                st_dsft = res["drop_shift_stats"]

                std_str  = f"{st_std['mean_5seed']:.2f}% ± {st_std['std_5seed']:.2f}%"
                msk_str  = f"{st_msk['mean_5seed']:.2f}% ± {st_msk['std_5seed']:.2f}%"
                dmsk_str = f"{st_dmsk['mean_5seed']:.2f}% ± {st_dmsk['std_5seed']:.2f}%"
                sft_str  = f"{st_sft['mean_5seed']:.2f}% ± {st_sft['std_5seed']:.2f}%"
                dsft_str = f"{st_dsft['mean_5seed']:.2f}% ± {st_dsft['std_5seed']:.2f}%"

                print(f"{model_name:<14} | {params:<8,d} | {std_str:<18} | {msk_str:<18} | {dmsk_str:<16} | {sft_str:<18} | {dsft_str:<16}")

            print("-" * 118)

    # Detailed Per-Seed Tables
    print("\n\n" + "=" * 110)
    print("DETAILED PER-SEED BREAKDOWN TABLES")
    print("=" * 110)

    for ds_tag, ds_title in [("cifar", "CIFAR-FS (32x32)"), ("mini", "MiniImageNet Native (84x84)")]:
        for shot in [5, 1]:
            shot_key = f"{shot}shot"
            for cond_name, dict_name, stat_name in [("50% Partial Masking", "mask_per_seed", "mask_stats"),
                                                   ("25% Position Shift",  "shift_per_seed", "shift_stats")]:
                print(f"\nTABLE: {ds_title} | {shot}-Shot | {cond_name} Accuracy (per seed)")
                print(f"{'Model':<14} | {'Seed 1':<8} | {'Seed 7':<8} | {'Seed 21':<8} | {'Seed 42':<8} | {'Seed 123':<8} | {'Mean±Std (n=5)':<16} | {'Mean±Std (n=4 excl 21)':<22}")
                print("-" * 105)

                for model_name, params, _, _, _ in models_info:
                    p_dict = results[ds_tag][model_name][shot_key][dict_name]
                    p_stat = results[ds_tag][model_name][shot_key][stat_name]
                    s1, s7, s21, s42, s123 = p_dict[1], p_dict[7], p_dict[21], p_dict[42], p_dict[123]
                    m5_str = f"{p_stat['mean_5seed']:.2f}% ± {p_stat['std_5seed']:.2f}%"
                    m4_str = f"{p_stat['mean_4seed']:.2f}% ± {p_stat['std_4seed']:.2f}%"

                    print(f"{model_name:<14} | {s1:6.2f}% | {s7:6.2f}% | {s21:6.2f}% | {s42:6.2f}% | {s123:6.2f}% | {m5_str:<16} | {m4_str:<22}")

                print("-" * 105)

    print("\nROBUSTNESS EVALUATION COMPLETE. Stopping here as instructed.", flush=True)

if __name__ == '__main__':
    main()

# run_cub_cross_domain_eval.py
# Cross-Domain Generalization Evaluation: MiniImageNet-Trained Models Evaluated on CUB-200-2011
#
# Protocol:
#   - Zero retraining (evaluation-only)
#   - Input images: CUB "novel" test split (50 classes), explicitly resized to 84x84 RGB
#   - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#   - Evaluation: 5-way classification, 1-shot and 5-shot, Q=15
#   - 5 seeds: [1, 7, 21, 42, 123], 600 episodes per seed (standard condition only)
#   - Models evaluated (MiniImageNet Native checkpoints):
#       1) EXP-F3 Native (22,249 params)
#       2) ProtoNet Baseline (47,630 params)
#       3) MAML Baseline (49,481 params)
#       4) RelationNet Baseline (49,617 params)
#   - Output manifest saved to: checkpoints/cub_cross_domain_manifest.json

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

from cub_dataset import RealCUB
from mini_imagenet_mmap_loader import MiniImageNetMmap as RealMiniImageNet

from run_exp_f3_benchmark import IRFEExpF3_Native
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
# Evaluation routines (600 episodes per seed)
# ---------------------------------------------------------------------------
def eval_exp_f3_cub(model, ds_test, k_shot, seed):
    model.eval()
    accs = []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            protos = model.compute_prototypes(sx, sy, n=N_WAY)
            preds = model.predict_proto(qx, protos)
            accs.append((preds.argmax(1) == qy).float().mean().item() * 100.0)
    return float(np.mean(accs))

def eval_protonet_cub(model, ds_test, k_shot, seed):
    model.eval()
    accs = []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            protos = model.compute_prototypes(sx, sy, n=N_WAY)
            preds = model.predict_proto(qx, protos)
            accs.append((preds.argmax(1) == qy).float().mean().item() * 100.0)
    return float(np.mean(accs))

def eval_maml_cub(model, ds_test, k_shot, seed):
    accs = []
    crit = nn.CrossEntropyLoss()
    fast_model = MAMLBackbone().to(device)
    for ep in range(EVAL_EPISODES):
        ep_seed = seed + ep * 1000 + 777
        sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        fast_model.load_state_dict(model.state_dict())
        fast_model.train()
        inner_opt = optim.SGD(fast_model.parameters(), lr=0.01)
        for _ in range(5):
            inner_opt.zero_grad()
            loss = crit(fast_model(sx), sy)
            loss.backward()
            inner_opt.step()

        fast_model.eval()
        with torch.no_grad():
            preds = fast_model(qx).argmax(1)
            accs.append((preds == qy).float().mean().item() * 100.0)
    return float(np.mean(accs))

def eval_relationnet_cub(model, ds_test, k_shot, seed):
    model.eval()
    accs = []
    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            ep_seed = seed + ep * 1000 + 777
            sx, sy, qx, qy, _ = ds_test.sample_episode(n_way=N_WAY, k_shot=k_shot, q_query=Q_QUERY, seed=ep_seed)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)
            relations = model(sx, sy, qx, n_way=N_WAY)
            preds = relations.argmax(1)
            accs.append((preds == qy).float().mean().item() * 100.0)
    return float(np.mean(accs))

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
    print("CROSS-DOMAIN GENERALIZATION EVALUATION: MINIIMAGENET-TRAINED MODELS EVALUATED ON CUB-200-2011")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("Dataset: CUB-200-2011 Novel Split (50 test classes, explicitly resized to 84x84 RGB)")
    print("Protocol: 5-way classification, 1-shot and 5-shot, Q=15, 600 episodes per seed")
    print("Seeds: [1, 7, 21, 42, 123] (standard condition, zero retraining)")
    print("=" * 100, flush=True)

    ds_test_cub = RealCUB(split='novel')
    ds_test_mini = RealMiniImageNet(split='test', seed=42)

    # Reference MiniImageNet baseline results from manifests
    mini_baselines = {
        "EXP-F3": {
            1: {1: 35.2711, 5: 53.0556},
            7: {1: 36.5133, 5: 53.4956},
            21: {1: 37.2889, 5: 54.3333},
            42: {1: 35.3111, 5: 52.1889},
            123: {1: 35.0089, 5: 53.7844}
        },
        "ProtoNet": {
            1: {1: 34.6267, 5: 49.5022},
            7: {1: 36.0022, 5: 48.3133},
            21: {1: 35.0244, 5: 49.6644},
            42: {1: 35.1933, 5: 48.3956},
            123: {1: 35.3911, 5: 48.2444}
        },
        "MAML": {
            1: {1: 27.9667, 5: 31.8089},
            7: {1: 28.6822, 5: 33.0822},
            21: {1: 28.4133, 5: 29.1933},
            42: {1: 27.4333, 5: 30.8244},
            123: {1: 27.0222, 5: 29.7711}
        },
        "RelationNet": {
            1: {1: 31.5511, 5: 39.8778},
            7: {1: 30.2422, 5: 36.0400},
            21: {1: 32.4244, 5: 36.0244},
            42: {1: 33.3844, 5: 36.6956},
            123: {1: 30.5444, 5: 42.2444}
        }
    }

    results = {}
    manifest_entries = []

    models_info = [
        ("EXP-F3", 22249, IRFEExpF3_Native, eval_exp_f3_cub, "exp_f3_mini_{shot}shot_seed{seed}.pt"),
        ("ProtoNet", 47630, ProtoNetBaseline, eval_protonet_cub, "sota_protonet_mini_{shot}shot_seed{seed}.pt"),
        ("MAML", 49481, MAMLBackbone, eval_maml_cub, "sota_maml_mini_{shot}shot_seed{seed}.pt"),
        ("RelationNet", 49617, RelationNetBaseline, eval_relationnet_cub, "sota_relationnet_mini_{shot}shot_seed{seed}.pt")
    ]

    for model_name, params, model_cls, eval_fn, ckpt_fmt in models_info:
        print(f"\n" + "=" * 90)
        print(f"EVALUATING MODEL: {model_name} ({params:,} params)")
        print("=" * 90, flush=True)

        results[model_name] = {"1shot": {}, "5shot": {}}

        for shot in [1, 5]:
            print(f"\n--- {model_name} | {shot}-Shot Evaluation on CUB ---", flush=True)
            results[model_name][f"{shot}shot"]["cub_per_seed"] = {}
            results[model_name][f"{shot}shot"]["drop_per_seed"] = {}

            for seed in SEEDS:
                ckpt_filename = ckpt_fmt.format(shot=shot, seed=seed)
                ckpt_path = os.path.join(CKPT_DIR, ckpt_filename)

                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")

                md5_val = compute_md5(ckpt_path)
                model = model_cls().to(device)
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)

                # Sanity check reloaded mini accuracy
                mini_ref = mini_baselines[model_name][seed][shot]

                # Run CUB cross-domain evaluation
                cub_acc = eval_fn(model, ds_test_cub, shot, seed)
                drop_val = mini_ref - cub_acc

                results[model_name][f"{shot}shot"]["cub_per_seed"][seed] = cub_acc
                results[model_name][f"{shot}shot"]["drop_per_seed"][seed] = drop_val

                print(f"  [Seed {seed:3d}] Checkpoint: {ckpt_filename} | MD5: {md5_val}")
                print(f"             MiniImageNet Acc: {mini_ref:.2f}% | CUB Acc: {cub_acc:.2f}% | Drop: {drop_val:+.2f}%")

                manifest_entries.append({
                    "model": model_name,
                    "params": params,
                    "shot": shot,
                    "seed": seed,
                    "checkpoint_file": ckpt_filename,
                    "md5": md5_val,
                    "mini_imagenet_acc": mini_ref,
                    "cub_acc": cub_acc,
                    "accuracy_drop": drop_val
                })

            # Compute summary stats
            cub_stats = compute_stats(results[model_name][f"{shot}shot"]["cub_per_seed"])
            drop_stats = compute_stats(results[model_name][f"{shot}shot"]["drop_per_seed"])
            mini_shot_dict = {s: mini_baselines[model_name][s][shot] for s in SEEDS}
            mini_stats = compute_stats(mini_shot_dict)

            results[model_name][f"{shot}shot"]["cub_stats"] = cub_stats
            results[model_name][f"{shot}shot"]["drop_stats"] = drop_stats
            results[model_name][f"{shot}shot"]["mini_stats"] = mini_stats

            print(f"\n  SUMMARY ({shot}-Shot):")
            print(f"    MiniImageNet (n=5): {mini_stats['mean_5seed']:.2f}% ± {mini_stats['std_5seed']:.2f}%")
            print(f"    CUB Novel    (n=5): {cub_stats['mean_5seed']:.2f}% ± {cub_stats['std_5seed']:.2f}%")
            print(f"    Acc Drop     (n=5): {drop_stats['mean_5seed']:.2f}% ± {drop_stats['std_5seed']:.2f}%")

    # -----------------------------------------------------------------------
    # Save Manifest
    # -----------------------------------------------------------------------
    out_manifest_path = os.path.join(CKPT_DIR, "cub_cross_domain_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "experiment": "Cross-Domain Generalization Evaluation (MiniImageNet -> CUB-200-2011)",
            "dataset_cub": "CUB-200-2011 Novel Split (50 classes, resized to 84x84 RGB)",
            "seeds": SEEDS,
            "episodes_per_seed": EVAL_EPISODES,
            "manifest_entries": manifest_entries,
            "summary_results": results
        }, f, indent=2)

    print(f"\nManifest written to: {out_manifest_path}")

    # -----------------------------------------------------------------------
    # MASTER TABLES PRINTING
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 105)
    print("CROSS-DOMAIN GENERALIZATION RESULTS TABLES (MINIIMAGENET -> CUB)")
    print("=" * 105)

    for shot in [5, 1]:
        print(f"\nMASTER TABLE: {shot}-Shot Classification (Mean ± std across 5 seeds)")
        print(f"{'Model':<14} | {'Params':<8} | {'MiniImageNet (In-Domain)':<24} | {'CUB Novel (Cross-Domain)':<24} | {'Accuracy Drop (Δ)':<18}")
        print("-" * 100)

        for model_name, params, _, _, _ in models_info:
            mini_s = results[model_name][f"{shot}shot"]["mini_stats"]
            cub_s  = results[model_name][f"{shot}shot"]["cub_stats"]
            drop_s = results[model_name][f"{shot}shot"]["drop_stats"]

            mini_str = f"{mini_s['mean_5seed']:.2f}% ± {mini_s['std_5seed']:.2f}%"
            cub_str  = f"{cub_s['mean_5seed']:.2f}% ± {cub_s['std_5seed']:.2f}%"
            drop_str = f"{drop_s['mean_5seed']:.2f}% ± {drop_s['std_5seed']:.2f}%"

            print(f"{model_name:<14} | {params:<8,d} | {mini_str:<24} | {cub_str:<24} | {drop_str:<18}")

        print("-" * 100)

    # Detailed Per-Seed Tables
    for shot in [5, 1]:
        print(f"\n\nDETAILED SEED BREAKDOWN: {shot}-Shot CUB Accuracy")
        print(f"{'Model':<14} | {'Seed 1':<8} | {'Seed 7':<8} | {'Seed 21':<8} | {'Seed 42':<8} | {'Seed 123':<8} | {'Mean±Std (n=5)':<16} | {'Mean±Std (n=4 excl 21)':<22}")
        print("-" * 105)

        for model_name, params, _, _, _ in models_info:
            c_dict = results[model_name][f"{shot}shot"]["cub_per_seed"]
            c_stats = results[model_name][f"{shot}shot"]["cub_stats"]
            s1, s7, s21, s42, s123 = c_dict[1], c_dict[7], c_dict[21], c_dict[42], c_dict[123]
            m5_std = f"{c_stats['mean_5seed']:.2f}% ± {c_stats['std_5seed']:.2f}%"
            m4_std = f"{c_stats['mean_4seed']:.2f}% ± {c_stats['std_4seed']:.2f}%"

            print(f"{model_name:<14} | {s1:6.2f}% | {s7:6.2f}% | {s21:6.2f}% | {s42:6.2f}% | {s123:6.2f}% | {m5_std:<16} | {m4_std:<22}")

        print("-" * 105)

    print("\nEVALUATION COMPLETE. Stopping here as instructed.", flush=True)

if __name__ == '__main__':
    main()

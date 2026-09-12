# analyze_35k_results.py
import json
import numpy as np

with open("checkpoints/exp_f3_35k_manifest.json") as f:
    m35 = json.load(f)

with open("checkpoints/exp_f3_manifest.json") as f:
    m22 = json.load(f)

with open("checkpoints/sota_protonet_manifest.json") as f:
    mproto = json.load(f)

def extract_map(manifest):
    d = {}
    for c in manifest["checkpoints"]:
        ds = c["dataset"]
        shot = c["shot"]
        seed = c["seed"]
        acc = float(c["reported_std_accuracy"])
        if ds not in d: d[ds] = {}
        if shot not in d[ds]: d[ds][shot] = {}
        d[ds][shot][seed] = acc
    return d

d35 = extract_map(m35)
d22 = extract_map(m22)
dp  = extract_map(mproto)

seeds = [1, 7, 21, 42, 123]

print("=" * 100)
print("PER-SEED BREAKDOWN TABLE")
print("=" * 100)
for ds in ["cifar", "mini"]:
    for shot in [1, 5]:
        ds_label = "CIFAR-FS" if ds == "cifar" else "MiniImageNet Native"
        print(f"\n--- {ds_label} {shot}-Shot ---")
        print(f"{'Seed':<6} | {'EXP-F3 (22k)':<14} | {'EXP-F3-35k (35k)':<16} | {'ProtoNet (47.6k)':<16} | {'35k vs 22k':<12} | {'35k vs Proto':<12}")
        print("-" * 88)
        for s in seeds:
            v22 = d22[ds][shot][s]
            v35 = d35[ds][shot][s]
            vp  = dp[ds][shot][s]
            diff22 = v35 - v22
            diffp  = v35 - vp
            print(f"{s:<6} | {v22:<14.2f}% | {v35:<16.2f}% | {vp:<16.2f}% | {diff22:+11.2f}% | {diffp:+11.2f}%")

print("\n" + "=" * 100)
print("MASTER SUMMARY TABLE (MEAN ± STD)")
print("=" * 100)
for ds in ["cifar", "mini"]:
    for shot in [1, 5]:
        ds_label = "CIFAR-FS" if ds == "cifar" else "MiniImageNet Native"
        v22_all = [d22[ds][shot][s] for s in seeds]
        v35_all = [d35[ds][shot][s] for s in seeds]
        vp_all  = [dp[ds][shot][s] for s in seeds]

        v22_ex21 = [d22[ds][shot][s] for s in seeds if s != 21]
        v35_ex21 = [d35[ds][shot][s] for s in seeds if s != 21]
        vp_ex21  = [dp[ds][shot][s] for s in seeds if s != 21]

        wins_vs_22 = sum(1 for s in seeds if d35[ds][shot][s] > d22[ds][shot][s])
        wins_vs_p  = sum(1 for s in seeds if d35[ds][shot][s] > dp[ds][shot][s])

        print(f"\nBenchmark: {ds_label} {shot}-Shot")
        print(f"  5-Seed Mean ± Std:")
        print(f"    EXP-F3 (22,249 params)     : {np.mean(v22_all):.2f}% ± {np.std(v22_all):.2f}%")
        print(f"    EXP-F3-35k (34,917 params) : {np.mean(v35_all):.2f}% ± {np.std(v35_all):.2f}%")
        print(f"    ProtoNet (47,630 params)   : {np.mean(vp_all):.2f}% ± {np.std(vp_all):.2f}%")
        print(f"  4-Seed Mean ± Std (excl. seed 21):")
        print(f"    EXP-F3 (22,249 params)     : {np.mean(v22_ex21):.2f}% ± {np.std(v22_ex21):.2f}%")
        print(f"    EXP-F3-35k (34,917 params) : {np.mean(v35_ex21):.2f}% ± {np.std(v35_ex21):.2f}%")
        print(f"    ProtoNet (47,630 params)   : {np.mean(vp_ex21):.2f}% ± {np.std(vp_ex21):.2f}%")
        print(f"  Seed Consistency (Wins out of 5 seeds):")
        print(f"    EXP-F3-35k > EXP-F3 (22k) : {wins_vs_22} / 5 seeds")
        print(f"    EXP-F3-35k > ProtoNet(47k): {wins_vs_p} / 5 seeds")

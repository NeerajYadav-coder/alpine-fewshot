# analyze_large_results.py
import json
import numpy as np

with open("checkpoints/exp_f3_large_manifest.json") as f: m_large = json.load(f)
with open("checkpoints/exp_f3_35k_manifest.json") as f: m_35k = json.load(f)
with open("checkpoints/exp_f3_manifest.json") as f: m_22k = json.load(f)
with open("checkpoints/sota_protonet_manifest.json") as f: m_proto = json.load(f)

def extract_map(manifest):
    d = {}
    for c in manifest["checkpoints"]:
        ds, shot, seed, acc = c["dataset"], c["shot"], c["seed"], float(c["reported_std_accuracy"])
        if ds not in d: d[ds] = {}
        if shot not in d[ds]: d[ds][shot] = {}
        d[ds][shot][seed] = acc
    return d

dl, d35, d22, dp = extract_map(m_large), extract_map(m_35k), extract_map(m_22k), extract_map(m_proto)
seeds = [1, 7, 21, 42, 123]

print("=" * 110)
print("EXP-F3-LARGE (122,897 PARAMS) PER-SEED BREAKDOWN TABLE")
print("=" * 110)

for ds in ["cifar", "mini"]:
    for shot in [1, 5]:
        ds_label = "CIFAR-FS" if ds == "cifar" else "MiniImageNet Native"
        print(f"\n--- {ds_label} {shot}-Shot ---")
        print(f"{'Seed':<6} | {'EXP-F3 (22k)':<14} | {'EXP-F3-35k (35k)':<16} | {'EXP-F3-Large (123k)':<18} | {'ProtoNet (47.6k)':<16} | {'Large vs 22k':<12} | {'Large vs Proto':<12}")
        print("-" * 115)
        for s in seeds:
            v22, v35, vl, vp = d22[ds][shot][s], d35[ds][shot][s], dl[ds][shot][s], dp[ds][shot][s]
            print(f"{s:<6} | {v22:<14.2f}% | {v35:<16.2f}% | {vl:<18.2f}% | {vp:<16.2f}% | {vl-v22:+11.2f}% | {vl-vp:+11.2f}%")

print("\n" + "=" * 110)
print("MASTER SUMMARY TABLE (MEAN ± STD)")
print("=" * 110)
for ds in ["cifar", "mini"]:
    for shot in [1, 5]:
        ds_label = "CIFAR-FS" if ds == "cifar" else "MiniImageNet Native"
        v22_all = [d22[ds][shot][s] for s in seeds]
        v35_all = [d35[ds][shot][s] for s in seeds]
        vl_all  = [dl[ds][shot][s] for s in seeds]
        vp_all  = [dp[ds][shot][s] for s in seeds]

        v22_ex21 = [d22[ds][shot][s] for s in seeds if s != 21]
        v35_ex21 = [d35[ds][shot][s] for s in seeds if s != 21]
        vl_ex21  = [dl[ds][shot][s] for s in seeds if s != 21]
        vp_ex21  = [dp[ds][shot][s] for s in seeds if s != 21]

        wins_vs_22 = sum(1 for s in seeds if dl[ds][shot][s] > d22[ds][shot][s])
        wins_vs_p  = sum(1 for s in seeds if dl[ds][shot][s] > dp[ds][shot][s])

        print(f"\nBenchmark: {ds_label} {shot}-Shot")
        print(f"  5-Seed Mean ± Std:")
        print(f"    EXP-F3 (22,249 params)       : {np.mean(v22_all):.2f}% ± {np.std(v22_all):.2f}%")
        print(f"    EXP-F3-35k (34,917 params)   : {np.mean(v35_all):.2f}% ± {np.std(v35_all):.2f}%")
        print(f"    EXP-F3-Large (122,897 params): {np.mean(vl_all):.2f}% ± {np.std(vl_all):.2f}%")
        print(f"    ProtoNet (47,630 params)     : {np.mean(vp_all):.2f}% ± {np.std(vp_all):.2f}%")
        print(f"  4-Seed Mean ± Std (excl. seed 21):")
        print(f"    EXP-F3 (22,249 params)       : {np.mean(v22_ex21):.2f}% ± {np.std(v22_ex21):.2f}%")
        print(f"    EXP-F3-35k (34,917 params)   : {np.mean(v35_ex21):.2f}% ± {np.std(v35_ex21):.2f}%")
        print(f"    EXP-F3-Large (122,897 params): {np.mean(vl_ex21):.2f}% ± {np.std(vl_ex21):.2f}%")
        print(f"    ProtoNet (47,630 params)     : {np.mean(vp_ex21):.2f}% ± {np.std(vp_ex21):.2f}%")
        print(f"  Seed Consistency (Wins out of 5 seeds):")
        print(f"    EXP-F3-Large > EXP-F3 (22k)   : {wins_vs_22} / 5 seeds")
        print(f"    EXP-F3-Large > ProtoNet(47k)  : {wins_vs_p} / 5 seeds")

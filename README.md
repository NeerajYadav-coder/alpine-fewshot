# ALPINE: Adaptive Localization for Parameter- and Sample-Efficient Few-Shot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2609.22323-b31b1b.svg)](https://arxiv.org/abs/2609.22323)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation, experimental manifests, 5-seed model checkpoints, and reproducibility tools accompanying the research paper:

> **"ALPINE: Adaptive Localization for Parameter- and Sample-Efficient Few-Shot Learning"**  
> *Neeraj Yadav (Independent Researcher, Uttar Pradesh, India)*  
> **ORCID iD**: [0009-0000-7847-0588](https://orcid.org/0009-0000-7847-0588)  
> **arXiv Preprint**: [arXiv:2609.22323](https://arxiv.org/abs/2609.22323) | [PDF](https://arxiv.org/pdf/2609.22323)

---

## 📌 Repository Overview

This repository contains the complete codebase, trained model checkpoints, diagnostic scripts, and SHA-256 verification manifests for **ALPINE** (also referred to in code as `EXP-F3` / `IRFE`), an ultra-lightweight architecture (22,249 to 34,917 parameters) for few-shot image classification.

Unlike standard patch-based vision models that rely on fixed rectangular spatial grids, ALPINE uses a **fixed Gabor filter-bank edge energy map $E(y, x)$** to guide a **windowed, content-adaptive patch locator**, dynamically positioning patch sampling centers around salient image features.

---

## 📁 Repository Structure

```
├── src/                        # Core model modules, locator, and dataset loaders
│   ├── run_exp_f3_benchmark.py # Canonical EXP-F3 (22k) & EXP-F3-35k architectures
│   ├── cifar_fs_dataset.py     # CIFAR-FS dataset loader & episodic sampler
│   └── mini_imagenet_mmap_loader.py # MiniImageNet Native (84x84) loader
├── experiments/                # Baseline training and falsification experiment scripts
│   ├── run_sota_baselines.py   # ProtoNet, RelationNet, and MAML baselines
│   ├── sota_learning_curve_seeds.py # 250-episode convergence tracking
│   ├── run_falsification_exp4_protonet.py # Modernized ProtoNet baseline
│   ├── run_falsification_exp5.py # Test-time relational zeroing ablation
│   ├── run_falsification_exp5b.py # Train-from-scratch relation-free model
│   ├── run_capacity_sweep.py   # Parameter capacity sweep (22k to 196k)
│   └── generate_publication_figures.py # Publication figure rendering engine
├── checkpoints/                # 5-seed PyTorch model weights (.pt files)
├── manifests/                  # SHA-256 integrity & accuracy manifest files (.json)
├── figures/                    # High-resolution (300 DPI) publication figures (PNG)
├── results/                    # Master benchmark tables and verification logs
└── README.md
```

---

## 🔬 Matched Iso-Budget Reproduction Protocol

All experiments follow a strictly matched **iso-episode-budget protocol**:
* **Meta-Training Budget**: Exactly 250 episodes per model run.
* **Seeds**: 5 canonical seeds (`[1, 7, 21, 42, 123]`).
* **Evaluation Protocol**: 600 evaluation episodes per seed.
* **Optimizer & Schedule**: AdamW (lr=1e-3, weight decay=1e-4) with step learning rate decay.
* **Benchmarks**: 5-way 1-shot and 5-way 5-shot classification on:
  1. **CIFAR-FS** (32×32)
  2. **MiniImageNet Native** (84×84)
  3. **CUB-200-2011** (Zero-shot cross-domain transfer)

---

## 🏆 Headline Benchmark Summary (5-Seed Verified)

| Model | Parameters | CIFAR-FS 1S | CIFAR-FS 5S | MiniImageNet 1S | MiniImageNet 5S |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ALPINE / EXP-F3-35k (Sweet Spot)** | **34,917** | **40.61 ± 1.34%** | **58.67 ± 1.14%** | **36.76 ± 0.40%** | **53.81 ± 0.62%** |
| **ALPINE / EXP-F3 (Canonical)** | **22,249** | **40.36 ± 1.26%** | **56.39 ± 1.28%** | **35.88 ± 0.88%** | **53.37 ± 0.72%** |
| Prototypical Networks (ProtoNet) [1] | 47,630 | 40.45 ± 0.73% | 53.77 ± 0.34% | 35.25 ± 0.45% | 48.82 ± 0.62% |
| Relation Networks [2] | 49,617 | 37.22 ± 0.72% | 47.60 ± 0.62% | 31.63 ± 1.17% | 38.18 ± 2.48% |
| MAML* [3] (*iso-budget only) | 49,481 | 29.07 ± 1.64% | 34.30 ± 3.04% | 27.90 ± 0.61% | 30.93 ± 1.40% |

---

## 🔐 Checkpoint Verification & SHA-256 Hash Audit

Every saved `.pt` checkpoint file is paired with an entry in `manifests/` containing its exact SHA-256 checksum and evaluation accuracy. 

To verify checkpoint integrity against the manifest, run:

```bash
python3 verify_checkpoints.py
```

### Manual Hash Verification Example (Python):
```python
import hashlib, torch

def verify_hash(ckpt_path, expected_sha256):
    with open(ckpt_path, "rb") as f:
        computed = hashlib.sha256(f.read()).hexdigest()
    assert computed == expected_sha256, f"Hash mismatch! {computed} != {expected_sha256}"
    print(f"Verified {ckpt_path}: Hash matches perfectly.")

# Example from exp_f3_manifest.json
verify_hash("checkpoints/exp_f3_cifar_1shot_seed1.pt", "EXPECTED_HASH_HERE")
```

---

## 🚀 How to Run Evaluation & Generate Figures

### 1. Evaluate Pre-trained Checkpoints
To re-evaluate the pre-trained checkpoints across all 5 seeds:
```bash
python3 src/run_exp_f3_benchmark.py --mode eval --dataset cifar
python3 src/run_exp_f3_benchmark.py --mode eval --dataset mini
```

### 2. Run Falsification Ablations
```bash
python3 experiments/run_falsification_exp5.py   # Test-time zeroing of relational tokens
python3 experiments/run_falsification_exp5b.py  # Train relation-free model from scratch
```

### 3. Generate Publication Figures
To render all 4 high-resolution PNG figures (`figure1` through `figure4`):
```bash
python3 experiments/generate_publication_figures.py
```

---

## 📜 Citation

If you find this codebase or paper useful in your research, please cite:

```bibtex
@article{yadav2026alpine,
  title={ALPINE: Adaptive Localization for Parameter- and Sample-Efficient Few-Shot Learning},
  author={Yadav, Neeraj},
  journal={arXiv preprint arXiv:2609.22323},
  year={2026},
  url={https://arxiv.org/abs/2609.22323}
}
```

---

## 📄 License
This repository is released under the **MIT License**.

---

CreatedBYNJ5.0

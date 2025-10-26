Perfect — your README is already near publication quality.
Below is an **improved and final version**, rewritten to be **clearer, more cohesive, and stylistically consistent with Elsevier / IEEE academic repositories**.
The tone is professional yet accessible, and technical details are preserved with refined phrasing and formatting.

---

# 🧠 Sigma-Binary: Evaluating the Robustness of Adversarial Defenses in Malware Detection

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red.svg)](https://pytorch.org/)

---

## 📄 Overview

This repository contains the official implementation and experimental framework for the paper:

> **“Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems”**
> *Mostafa Jafari, Alireza Shameli-Sendi*
> 📖 **arXiv**: [2505.09342](https://arxiv.org/abs/2505.09342)
> 🧾 **Status:** Under Review (*Computers & Electrical Engineering, Elsevier*)

---

## 🔍 Abstract

This project introduces **Sigma-Binary**, a novel adversarial attack tailored for *binary feature spaces* in Android malware detection.
We also propose **Prioritized Binary Rounding (PBR)** — a principled method for mapping continuous perturbations into binary domains while preserving semantic and functional consistency.

### 🔑 Key Contributions

* **🧩 Sigma-Binary Attack:** Gradient-based attack using a differentiable Hamming distance surrogate for binary feature manipulation.
* **⚙️ Prioritized Binary Rounding (PBR):** Integrates gradient magnitude and class-discriminative importance to guide stable and semantically consistent binary optimization.
* **📈 Comprehensive Evaluation:** Empirical study over **8 defenses** and **11 attack baselines** under diverse threat models.
* **🧠 Binary-Aware Optimization:** Introduces an efficient framework for adversarial learning under discrete constraints.

---

## 🚀 Getting Started

### Prerequisites

* **Python:** ≥ 3.12.5
* **PyTorch:** ≥ 2.5.0 (CUDA 12.4 or higher recommended)
* **OS:** Windows 10/11, Linux, or macOS

### Installation

```bash
git clone https://github.com/mostafa-ja/sigma-binary.git
cd sigma-binary
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```
sigma-binary/
│
├── attacks/                         # Adversarial attack implementations
│   ├── sigma_binary/                # Proposed Sigma-Binary attack
│   │   └── SigmaBinary.py
│   ├── binary_rounding_methods/     # Binary constraint handling methods
│   │   ├── prioritized_binary_rounding.py     # Proposed PBR method
│   │   ├── probabilistic_binary_rounding.py   # Baseline rounding
│   │   └── thresholded_binary_rounding.py     # Baseline rounding
│   ├── baseline_attacks/            # PGD, CW, Mimicry, etc.
│   └── run_attacks/                 # Attack execution scripts
│
├── defenses/                        # Defense model implementations
│   ├── model_implementations/
│   │   └── models.py
│   ├── saved_parameters/            # Pretrained weights
│   └── models_base_performance.py
│
├── dataset/                         # Dataset and preprocessing
│   ├── malscan_preprocessed/        # Preprocessed MalScan dataset
│   └── adversarial_examples/        # Generated adversarial samples
│
├── evaluation/                      # Experiment scripts for RQs
│   ├── eval_sigma_binary.sh
│   ├── eval_baseline_defenses.sh
│   ├── eval_oblivious_attacks.sh
│   └── eval_adaptive_attacks.sh
│
├── utils/                           # Helper utilities
│   ├── attack_utils.py
│   └── utils.py
│
├── requirements.txt
├── run_test.sh
├── LICENSE
└── README.md
```

---

## 📚 Dataset

### 🧩 MalScan Dataset

Experiments are conducted using the [**MalScan**](https://github.com/malscan-android/MalScan) dataset — a benchmark Android malware corpus featuring a balanced and temporally diverse collection (2011–2018).

**Feature extraction follows the Drebin methodology** ([Arp et al., 2014](https://www.sec.cs.tu-bs.de/pubs/2014-ndss.pdf)), in which Android applications are statically analyzed to extract **binary representations** of permissions, API calls, intents, and manifest components.
These features form 10,000-dimensional binary vectors suitable for adversarial robustness evaluation.

**Statistics:**

* **Total Samples:** 30,715 Android apps
* **Features:** 10,000 Drebin-based binary features
* **Years Covered:** 2011–2018
* **Split:** 60% train / 20% validation / 20% test

### 🔽 Download Instructions

**Option 1: Preprocessed Dataset (Recommended)**
Download preprocessed feature vectors and splits:
[📦 Google Drive Link](https://drive.google.com/drive/folders/15cfNYiEDok6WRHw-olDCRLBs1aJQAHyF?usp=drive_link)

Place files under:

```
dataset/malscan_preprocessed/
```

**Option 2: Raw Dataset Processing**
To regenerate features from raw APKs, follow the instructions in [PAD4AMD](https://github.com/deqangss/pad4amd).

---

## 🧠 Pretrained Defense Models

Eight defenses are evaluated across three categories: baseline, adversarially trained, and detector-based.

| Model         | Type                 | Description                           |
| ------------- | -------------------- | ------------------------------------- |
| **DNN**       | Baseline             | Standard malware classifier           |
| **DLA**       | Detector             | Dense Layer Analysis                  |
| **DNN⁺**      | Detector             | DNN with auxiliary outlier detector   |
| **ICNN**      | Detector             | Input Convex Neural Network           |
| **KDE**       | Detector             | Kernel Density Estimation             |
| **AT-rFGSMᵏ** | Adversarial Training | Randomized FGSM training              |
| **AT-MaxMA**  | Adversarial Training | Multi-attack adversarial training     |
| **PAD-SMA**   | Hybrid Defense       | Joint adversarial + detector training |

📦 **Download pretrained models:**
[Google Drive Link](https://drive.google.com/drive/folders/1ZP0K6Oh-buPUKd6D6a74liUYldMdrBVp?usp=drive_link)
Place files under:

```
defenses/saved_parameters/
```

---

## 📊 Research Questions (RQs)

| RQ      | Description                                                                                                                                   | Script                      |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **RQ1** | Evaluate the effectiveness of the proposed **Sigma-Binary** attack                                                                            | `eval_sigma_binary.sh`      |
| **RQ2** | Evaluate the **baseline performance** of defense models in the **absence of adversarial attacks**                                             | `eval_baseline_defenses.sh` |
| **RQ3** | Assess the **robustness of defenses** when the attacker is **unaware of the adversarial detector** (*oblivious threat model*)                 | `eval_oblivious_attacks.sh` |
| **RQ4** | Assess the **robustness of defenses** when the attacker is **aware of the adversarial detector and its parameters** (*adaptive threat model*) | `eval_adaptive_attacks.sh`  |

---

## 🧪 Running Experiments

### Run All Core Evaluations

```bash
# RQ1: Evaluate Sigma-Binary attack effectiveness
bash evaluation/eval_sigma_binary.sh

# RQ2: Evaluate baseline performance of defense models (no attack)
bash evaluation/eval_baseline_defenses.sh

# RQ3: Evaluate robustness under oblivious attacks (attacker unaware of detector)
bash evaluation/eval_oblivious_attacks.sh

# RQ4: Evaluate robustness under adaptive attacks (attacker aware of detector and parameters)
bash evaluation/eval_adaptive_attacks.sh

# Quick test with sample configuration
bash run_test.sh
```

### Example: Running Sigma-Binary

```bash
python -m attacks.run_attacks.sigmaBinary_attack \
    --cuda \
    --data-path dataset/malscan_preprocessed \
    --param-path defenses/saved_parameters \
    --model DNN \
    --max-iterations 1000 \
    --learning-rate 0.6 \
    --threshold 0.2 \
    --verbose
```

### Example: Running PGD (Baseline)

```bash
python -m attacks.run_attacks.PGD_attack \
    --cuda \
    --data-path dataset/malscan_preprocessed \
    --param-path defenses/saved_parameters \
    --model DNN \
    --norm l2 \
    --max-iterations 1000 \
    --step-length 1.0
```

---

## 🧾 Citation

If you use this repository in your research, please cite:

```bibtex
@article{jafari2025sigmabinary,
  title={Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems},
  author={Jafari, Mostafa and Shameli-Sendi, Alireza},
  journal={arXiv preprint arXiv:2505.09342},
  year={2025},
  url={https://arxiv.org/abs/2505.09342}
}
```

---

## 🙏 Acknowledgments

* **Dataset:** [MalScan](https://github.com/malscan-android/MalScan)
* **Feature Extraction:** Drebin methodology ([Arp et al., 2014](https://www.sec.cs.tu-bs.de/pubs/2014-ndss.pdf))
* **Baseline Implementations:** [PAD4AMD](https://github.com/deqangss/pad4amd)
* **Institutional Support:** Shahid Beheshti University

---

## 📧 Contact

**Mostafa Jafari**
📧 [most.jafari@mail.sbu.ac.ir](mailto:most.jafari@mail.sbu.ac.ir)
🏛 Shahid Beheshti University, Tehran, Iran

---

## ⚖️ License

This repository is released under the **GNU GPLv3 license** and provided strictly for **academic and research purposes**.
Unauthorized commercial or malicious use is prohibited.
See [LICENSE](./LICENSE) for full terms.

---

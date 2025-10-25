# Sigma-Binary: Evaluating the Robustness of Adversarial Defenses in Malware Detection

Welcome to the official repository for **"Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems"**, a research paper currently under review.

📄 **Paper**: The latest version is available on [arXiv](https://arxiv.org/abs/2505.09342).

This repository contains the code, datasets, and scripts required to reproduce the experiments and results presented in the paper.

---

## 📌 Overview

This project introduces and evaluates **Sigma-Binary**, a novel adversarial attack specifically designed for binary-constrained domains in malware detection. Key contributions include:

- **Prioritized Binary Rounding**: An efficient method for mapping continuous perturbations into binary space while retaining high attack success.
- **Sigma-Binary Attack**: A gradient-based attack using differentiable approximations of Hamming distance for binary feature manipulation in malware detection systems.

---

## 🚀 Dependencies

Developed and tested on **Windows** with the following dependencies:

```plaintext
Python 3.12.5
torch == 2.5.0+cu124
numpy == 2.1.2
scikit-learn == 1.5.2
scipy == 1.14.1
````

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```text
sigma-binary-attack/
│
├── attacks/                          # Adversarial attack modules
│   ├── sigma_binary/                 # Sigma-Binary implementation
│   ├── binary_rounding_methods/     # Binary rounding strategies
│   │   ├── prioritized_binary/       # Our proposed Prioritized Binary Rounding method
│   │   ├── probabilistic_binary/     # Probabilistic Binary Rounding method
│   │   ├── thresholded_binary/       # Thresholded Binary Rounding method
│   ├── baseline_attacks/            # Existing baseline attacks (CW, PGD, etc.)
│   ├── run_attacks/                 # Scripts to execute attacks
│
├── defenses/                         # Defense model implementations
│   ├── model_implementations/       # Source code for defenses
│   ├── saved_parameters/            # Pretrained defense model weights
│
├── dataset/                          # Dataset and preprocessing files
│   ├── malscan_preprocessed/        # Preprocessed Malscan dataset
│   ├── adversarial_examples/        # Generated adversarial samples
│
├── evaluation/                       # Scripts for evaluating attacks and defenses
│   ├── eval_sigma_binary.sh         # RQ1: Sigma-Binary attack
│   ├── eval_baseline_defenses.sh    # RQ2: Baseline defense evaluation
│   ├── eval_oblivious_attacks.sh    # RQ3: Oblivious attack evaluation
│   ├── eval_adaptive_attacks.sh     # RQ4: Adaptive attack evaluation
│
├── utils/                            # Utility functions
│   ├── attack_utils.py              # Helpers for attack modules
│   ├── model_utils.py               # Utilities for models and preprocessing
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── run_test.sh                       # Sample execution script
```

---

## 📊 Dataset

Experiments are conducted using the [Malscan](https://github.com/malscan-android/MalScan) dataset.

To reproduce our results:

1. Follow preprocessing instructions from the **PAD** paper: [PAD4AMD GitHub](https://github.com/deqangss/pad4amd)
2. Or download our preprocessed dataset and feature mask from [Google Drive](https://drive.google.com/drive/folders/1kPzuph_N4TmM3F4z7gj3qcbL_7uLZANI?usp=sharing)
3. Place the downloaded files in: `dataset/malscan_preprocessed/`

---

## 🎯 Pretrained Defense Models

Defense models were trained following the **PAD** methodology.

* Download pretrained parameters from: [Google Drive](https://drive.google.com/drive/folders/1-q3TMZGjoDpBkNHgc5tTcFKMF9ywowQy?usp=sharing)
* Place them in: `defenses/saved_parameters/`

---

## 🛠 Evaluation Scripts

Run evaluation scripts corresponding to each research question (RQ):

```bash
bash evaluation/eval_sigma_binary.sh      # RQ1: Evaluate Sigma-Binary attack
bash evaluation/eval_baseline_defenses.sh # RQ2: Evaluate baseline defenses
bash evaluation/eval_oblivious_attacks.sh # RQ3: Oblivious attack evaluation
bash evaluation/eval_adaptive_attacks.sh  # RQ4: Adaptive attack evaluation
```

---

### ▶️ Example: Running an Attack

Run the Sigma-Binary attack:

```bash
python -m attacks.run_attacks.sigma_zero_attack --cuda \
       --data-path dataset/malscan_preprocessed \
       --param-path defenses/saved_parameters \
       --model DNN \
       --max-iterations 1000 \
       --learning-rate 0.6 \
       --threshold 0.2 \
       --verbose
```

---

## 🔗 Acknowledgments

We build upon prior work and reuse selected components (e.g., model architecture, preprocessing, training procedures) from the [PAD4AMD repository](https://github.com/deqangss/pad4amd).

---

## 📖 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{jafari2025sigmabinary,
  title={Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems},
  author={Jafari, Mostafa and Shameli-Sendi, Alireza},
  journal={arXiv preprint arXiv:2505.09342},
  year={2025}
}
```

---

## 📧 Contact

For questions or feedback, feel free to reach out:

✉️ [most.jafari@mail.sbu.ac.ir](mailto:most.jafari@mail.sbu.ac.ir)

---


## ⚖️ License

- This code is provided for **educational and research purposes only**. Any misuse may result in legal consequences. The authors and their organization bear no responsibility for illegal activities.
- This project is released under the **GPL license** (see [LICENSE](./LICENSE)).


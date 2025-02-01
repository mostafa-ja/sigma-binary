# Sigma-Binary: Evaluating the Robustness of Adversarial Defenses in Malware Detection

This repository accompanies the paper **Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems**, which has been submitted to IEEE Transactions on Information Forensics & Security (T-IFS) and is currently under review.

## 📌 Overview

This repository provides an implementation and evaluation framework for **Sigma-Binary**, a novel adversarial attack designed for binary-constrained domains. Our research introduces:

- **Prioritized Binary Rounding**: A novel technique for efficiently mapping continuous adversarial perturbations to binary space while maintaining high attack success rates.
- **Sigma-Binary Attack**: A gradient-based attack optimized for binary malware detection models, leveraging differentiable approximations of the Hamming distance.

## 🚀 Dependencies

Our code is developed and tested on **Windows**. The required dependencies are:

```plaintext
Python 3.12.5
torch == 2.5.0+cu124
numpy == 2.1.2
scikit-learn == 1.5.2
scipy == 1.14.1
```

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## 📂 Repository Structure

```
sigma-binary-attack/
│── attacks/                          # Adversarial attack implementations
│   ├── sigma_binary/                 # Our proposed Sigma-Binary attack
│   ├── binary_rounding_methods/      # Binary rounding methods
│   │   ├── prioritized_binary        # Our proposed method
│   │   ├── probabilistic_binary      # Comparison method 1
│   │   ├── thresholded_binary        # Comparison method 2
│   ├── baseline_attacks/             # Other adversarial attacks (CW, PGD, etc.)
│   ├── run_attacks/                  # Scripts for running attacks
│
│── defenses/                         # Defense models
│   ├── model_implementations/        # Code for different defense models
│   ├── saved_parameters/             # Pre-trained defense model parameters
│
│── dataset/                          # Data and preprocessing
│   ├── malscan_preprocessed/         # Preprocessed Malscan dataset
│   ├── adversarial_examples/         # Generated adversarial examples
│
│── evaluation/                       # Scripts for evaluation
│   ├── eval_PBR.sh                   # Evaluates Prioritized Binary Rounding (RQ1)
│   ├── eval_sigma_binary.sh          # Evaluates Sigma-Binary attack (RQ2)
│   ├── eval_baseline_defenses.sh     # Evaluates baseline defense performance (RQ3)
│   ├── eval_oblivious_attacks.sh     # Evaluates defenses against oblivious attacks (RQ4)
│   ├── eval_adaptive_attacks.sh      # Evaluates defenses against adaptive attacks (RQ5)
│
│── utils/                            # Helper functions
│   ├── attack_utils.py               # Shared functions for attacks
│   ├── model_utils.py                # Utility functions for models & dataset preprocessing
│
│── README.md                         # Repository overview
│── requirements.txt                  # Dependencies
│── run_test.sh                        # Example script demonstrating usage
```

## 📊 Dataset

We conduct our experiments using the [Malscan](https://github.com/malscan-android/MalScan) dataset. To reproduce our results:

1. Follow the preprocessing steps from **"PAD: Towards Principled Adversarial Malware Detection"** ([GitHub](https://github.com/deqangss/pad4amd)).
2. Alternatively, download our preprocessed dataset and feature mask from [this Google Drive link](https://drive.google.com/drive/folders/1kPzuph_N4TmM3F4z7gj3qcbL_7uLZANI?usp=sharing).
3. Place the files under `dataset/malscan_preprocessed/`.

## 🎯 Learned Parameters

The defense models were trained following the **PAD** methodology. Pre-trained parameters are available for download:

- [Trained defense parameters](https://drive.google.com/drive/folders/1-q3TMZGjoDpBkNHgc5tTcFKMF9ywowQy?usp=sharing).
- Place the downloaded files in `defenses/saved_parameters/`.

## 🛠 Running Evaluation Scripts

Each evaluation script is designed to test different aspects of our study. They can be executed as follows:

```bash
bash evaluation/eval_PBR.sh            # Evaluates Prioritized Binary Rounding (RQ1)
bash evaluation/eval_sigma_binary.sh   # Evaluates Sigma-Binary attack (RQ2)
bash evaluation/eval_baseline_defenses.sh   # Evaluates baseline performance of defenses (RQ3)
bash evaluation/eval_oblivious_attacks.sh   # Evaluates defenses against oblivious attacks (RQ4)
bash evaluation/eval_adaptive_attacks.sh    # Evaluates defenses against adaptive attacks (RQ5)
```

### Example: Running an attack manually

```bash
python -m attacks.run_attacks.sigma_zero_attack --cuda \
       --data-path dataset/malscan_preprocessed \
       --param-path defenses/saved_parameters \
       --model DNN \
       --max-iterations 10000 \
       --learning-rate 0.5 \
       --threshold 0.1 \
       --verbose
```

To run the attack using Prioritized Binary Rounding:

```bash
python -m attacks.run_attacks.sigma_zero_attack --cuda \
       --data-path dataset/malscan_preprocessed \
       --param-path defenses/saved_parameters \
       --model DNN \
       --max-iterations 10000 \
       --learning-rate 0.5 \
       --threshold 0.1 \
       --rounded --rounding-function "prioritized_binary_rounding" \
       --verbose
```

## 🔗 Acknowledgment

We adapt parts of our code (models, preprocessing, learned parameters) from the [PAD4AMD repository](https://github.com/deqangss/pad4amd).

## 📖 Citation

If you find our work useful, please cite it as follows:

```bibtex

```

## 📧 Contact

For inquiries, feel free to reach out:

✉️ [most.jafari@mail.sbu.ac.ir](mailto:most.jafari@mail.sbu.ac.ir)

## ⚖️ License

- This code is provided for **educational and research purposes only**. Any misuse may result in legal consequences. The authors and their organization bear no responsibility for illegal activities.
- This project is released under the **GPL license** (see [LICENSE](./LICENSE)).


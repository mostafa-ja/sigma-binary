"""
Reproducible runner for the Sigma Zero attack.
Usage example:
python -m attacks.run_attacks.sigmaZero_attack \
  --cuda --data-path dataset/malscan_preprocessed \
  --param-path defenses/saved_parameters \
  --model DNN --max-iterations 1000 --learning-rate 0.5 --threshold 0.3 --verbose
"""

import torch
import argparse
import json
import os
import time
from pathlib import Path

from utils.utils import *
from defenses.model_implementations.models import *
from attacks.baseline_attacks.Sigma_zero import sigma_zero
from attacks.binary_rounding_methods.prioritized_binary_rounding import prioritized_binary_rounding
from attacks.binary_rounding_methods.probabilistic_binary_rounding import probabilistic_binary_rounding
from attacks.binary_rounding_methods.thresholded_binary_rounding import thresholded_binary_rounding


def main():
    parser = argparse.ArgumentParser(description="Run Sigma Zero attack (reproducible, GitHub-ready).")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True,
                        choices=["DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"])
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative",
                        help="Indicator masking mode")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack (default: adaptive)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default 1024)")
    parser.add_argument("--verbose", action="store_true", help="Print attack progress")

    # core attack hyperparameters
    parser.add_argument("--max-iterations", type=int, default=1000, help="Max iterations for Sigma Zero")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Learning rate (default 0.5)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold value (default 0.3)")
    parser.add_argument("--binary-search-steps", type=int, default=4, help="Binary search steps")
    parser.add_argument("--initial-const", type=float, default=0.1, help="Initial constant for loss balancing")

    # rounding options
    parser.add_argument("--rounded", action="store_true", help="Apply rounding to adversarial examples")
    parser.add_argument("--rounding-function", type=str,
                        choices=["prioritized_binary_rounding", "probabilistic_binary_rounding", "thresholded_binary_rounding"],
                        default="thresholded_binary_rounding", help="Rounding method")
    parser.add_argument("--rounding-args", type=str, default="{}", help="JSON args for rounding function")

    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples",
                        help="Directory to save adversarial examples")
    args = parser.parse_args()

    # device and reproducibility
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # parse rounding args
    try:
        rounding_args = json.loads(args.rounding_args)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON for --rounding-args (e.g. '{\"threshold\":0.5}')")

    rounding_map = {
        "prioritized_binary_rounding": prioritized_binary_rounding,
        "probabilistic_binary_rounding": probabilistic_binary_rounding,
        "thresholded_binary_rounding": thresholded_binary_rounding,
    }
    rounding_fn = rounding_map[args.rounding_function]

    # paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # record metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": str(device),
        "args": vars(args),
    }
    with open(output_dir / "run_sigma_zero_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # dataset and mask
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)

    # model loading
    model_map = {
        "DNN": lambda: MalwareDetectionDNN("DNN", device=device),
        "AT_rFGSM": lambda: MalwareDetectionDNN("AT_rFGSM", device=device),
        "AT_MaxMA": lambda: MalwareDetectionDNN("AT_MaxMA", device=device),
        "KDE": lambda: KernelDensityEstimation(MalwareDetectionDNN("DNN", device=device).float(), device=device, model_name="KDE"),
        "DLA": lambda: AMalwareDetectionDLA(MalwareDetectionDNN("DNN", device=device), device=device),
        "DNNPlus": lambda: AMalwareDetectionDNNPlus(MalwareDetectionDNN("DNN", device=device), device=device, model_name="DNNPlus"),
        "ICNN": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="ICNN"),
        "PAD": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="PAD"),
    }
    model = model_map[args.model]()
    model.load(args.param_path)
    model.to(device)

    indicator_masking = args.mode == "deferred"
    detector_enabled = args.model in ["KDE", "DNNPlus", "DLA", "ICNN", "PAD"]

    attack_name = "sigma_zero"
    mals_path = output_dir / f"{model.model_name}_mals_{attack_name}.pt"
    advs_path = output_dir / f"{model.model_name}_advs_{attack_name}.pt"

    # attack execution
    if mals_path.exists() and advs_path.exists():
        loaded = load_advs(model.model_name, attack_name, output_dir=str(output_dir))
        mals_adapt, advs_adapt = loaded["mals"], loaded["advs"]
    else:
        mals_adapt, advs_adapt = attack_mal_batches(
            test_loader, model, sigma_zero, feature_mask, device,
            detector_enabled=detector_enabled, oblivion=args.oblivion,
            indicator_masking=indicator_masking, verbose=args.verbose,
            max_iterations=args.max_iterations, learning_rate=args.learning_rate,
            threshold=args.threshold, binary_search_steps=args.binary_search_steps,
            initial_const=args.initial_const
        )

    # evaluation
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion,
        indicator_masking=indicator_masking, rounded=args.rounded,
        batch_size=args.batch_size, norm="L0",
        rounding_function=rounding_fn, rounding_args=rounding_args
    )

    # determine save suffix
    if not args.rounded:
        suffix = ""
    elif args.rounding_function == "prioritized_binary_rounding":
        suffix = "_R"
    elif args.rounding_function == "thresholded_binary_rounding":
        suffix = "_R_thresh"
    else:
        suffix = "_R_prob"

    save_advs(model_name=model.model_name,
              attack_name=f"{attack_name}{suffix}",
              output_dir=str(output_dir),
              mals=mals_adapt, advs=advs_adapt)

    print(f"[sigmaZero_attack] Finished. Results saved in {output_dir}")


if __name__ == "__main__":
    main()

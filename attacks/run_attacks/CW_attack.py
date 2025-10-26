"""
Reproducible runner for the Carlini & Wagner (CW) binary-feature adversarial attack.

Usage example:
python -m attacks.run_attacks.CW_attack \
  --cuda --data-path dataset/malscan_preprocessed \
  --param-path defenses/saved_parameters \
  --model DNN --max-iterations 1000 --learning-rate 0.5 --verbose
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch

from utils.utils import *
from defenses.model_implementations.models import *
from attacks.baseline_attacks.CW import CW


def main():
    parser = argparse.ArgumentParser(description="Run Carlini–Wagner (CW) attack (reproducible, GitHub-ready).")
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
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    # Attack hyperparameters
    parser.add_argument("--max-iterations", type=int, default=1000, help="Maximum optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Learning rate for CW optimization")
    parser.add_argument("--binary-search-steps-cw", type=int, default=8, help="Binary search steps for CW outer loop")
    parser.add_argument("--binary-search-steps-penalty", type=int, default=4, help="Binary search steps for penalty term")
    parser.add_argument("--initial-const-cw", type=float, default=1.0, help="Initial CW constant")
    parser.add_argument("--initial-const-penalty", type=float, default=1.0, help="Initial penalty constant")
    parser.add_argument("--primary-confidence", type=float, default=1e-4, help="Primary confidence bound")
    parser.add_argument("--secondary-confidence", type=float, default=1e-4, help="Secondary confidence bound")

    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples",
                        help="Directory to save adversarial examples")

    args = parser.parse_args()

    # Setup device and reproducibility
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Output directory and metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": str(device),
        "args": vars(args)
    }
    with open(output_dir / "run_CW_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Load dataset and feature mask
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)

    # Model mapping
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
    model.eval()

    indicator_masking = args.mode == "deferred"
    detector_enabled = args.model in ["KDE", "DNNPlus", "DLA", "ICNN", "PAD"]

    attack_name = "CW"
    file_paths = {
        "mals": output_dir / f"{model.model_name}_mals_{attack_name}.pt",
        "advs": output_dir / f"{model.model_name}_advs_{attack_name}.pt",
    }

    # Run or load cached results
    if file_paths["mals"].exists() and file_paths["advs"].exists():
        loaded = load_advs(model.model_name, attack_name, output_dir=str(output_dir))
        mals_adapt, advs_adapt = loaded["mals"], loaded["advs"]
    else:
        mals_adapt, advs_adapt = attack_mal_batches(
            test_loader, model, CW, feature_mask, device,
            detector_enabled=detector_enabled, oblivion=args.oblivion,
            indicator_masking=indicator_masking, verbose=args.verbose,
            max_iterations=args.max_iterations, learning_rate=args.learning_rate,
            binary_search_steps_cw=args.binary_search_steps_cw,
            binary_search_steps_penalty=args.binary_search_steps_penalty,
            initial_const_cw=args.initial_const_cw, initial_const_penalty=args.initial_const_penalty,
            primary_confidence=args.primary_confidence, secondary_confidence=args.secondary_confidence
        )

    # Evaluate adversarial examples
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion, indicator_masking=indicator_masking,
        rounded=False, batch_size=args.batch_size, norm="L0"
    )

    # Save results
    save_advs(
        model_name=model.model_name,
        attack_name=attack_name,
        output_dir=str(output_dir),
        mals=mals_adapt,
        advs=advs_adapt,
    )

    print(f"[CW_attack] Finished. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

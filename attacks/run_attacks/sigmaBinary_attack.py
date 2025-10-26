"""
Reproducible runner for the Sigma Binary.
Usage example:
python -m attacks.run_attacks.sigmaBinary_attack \
--cuda --data-path dataset/malscan_preprocessed \
--param-path defenses/saved_parameters \
--model DNN --max-iterations 1000 --learning-rate 0.5 --threshold 0.2 --verbose
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from utils.utils import *
from defenses.model_implementations.models import *
from attacks.sigma_binary.SigmaBinary import sigmaBinary  # explicit import for clarity


def build_model(model_name: str, device: torch.device):
    mapping = {
        "DNN": lambda: MalwareDetectionDNN("DNN", device=device),
        "AT_rFGSM": lambda: MalwareDetectionDNN("AT_rFGSM", device=device),
        "AT_MaxMA": lambda: MalwareDetectionDNN("AT_MaxMA", device=device),
        "KDE": lambda: KernelDensityEstimation(MalwareDetectionDNN("DNN", device=device).float(), device=device, model_name="KDE"),
        "DLA": lambda: AMalwareDetectionDLA(MalwareDetectionDNN("DNN", device=device), device=device),
        "DNNPlus": lambda: AMalwareDetectionDNNPlus(MalwareDetectionDNN("DNN", device=device), device=device, model_name="DNNPlus"),
        "ICNN": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="ICNN"),
        "PAD": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="PAD"),
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    return mapping[model_name]()


def main():
    parser = argparse.ArgumentParser(description="Run sigmaBinary attack (reproducible, repo-ready).")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True, choices=[
        "DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"
    ], help="Model name to attack")
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative",
                        help="Indicator masking mode: 'deferred' (True) or 'conservative' (False)")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack (default: adaptive attack)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for loading the dataset (default: 1024)")
    parser.add_argument("--verbose", action="store_true", help="Print attack process details")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error prints (overrides --verbose)")

    # Attack hyperparams (defaults chosen to match sigmaBinary defaults)
    parser.add_argument("--max-iterations", type=int, default=1000, help="Maximum iterations for Sigma Binary attack")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Learning rate for the attack (default 0.5)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold value for the attack (default 0.3)")
    parser.add_argument("--sigma", type=float, default=1e-4, help="Sigma parameter for L0 approximation")
    parser.add_argument("--binary-search-steps", type=int, default=4, help="Number of binary search steps for the attack")
    parser.add_argument("--initial-const", type=float, default=0.1, help="Initial constant for binary search")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples",
                        help="Directory to save adversarial example tensors (default: dataset/adversarial_examples)")
    parser.add_argument('--confidence-bound-primary', type=float, default=0.4, help='Primary confidence bound')
    parser.add_argument('--confidence-bound-secondary', type=float, default=0.4, help='Secondary confidence bound')
    parser.add_argument('--confidence-update-interval', type=int, default=80, help='Confidence update interval')

    args = parser.parse_args()

    # Configure logging/printing
    def log(*a, **k):
        if not args.quiet:
            print(*a, **k)

    # Device & reproducibility
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    log(f"[run_attack] device = {device}, seed = {args.seed}")

    # Validate paths
    data_path = Path(args.data_path)
    param_path = Path(args.param_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"[ERROR] data-path does not exist: {data_path}", file=sys.stderr)
        sys.exit(2)
    if not param_path.exists():
        print(f"[ERROR] param-path does not exist: {param_path}", file=sys.stderr)
        sys.exit(2)

    # save run metadata for reproducibility
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "args": vars(args),
        "device": str(device)
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # Load dataset
    _, _, test_loader = load_data(str(data_path), args.batch_size)
    feature_mask = load_feature_mask(str(data_path), device)

    # Build and load model
    model = build_model(args.model, device)
    try:
        model.load(str(param_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model parameters from {param_path}: {e}", file=sys.stderr)
        sys.exit(3)
    model.to(device)
    model.eval()

    # Attack configuration
    indicator_masking = args.mode == "deferred"
    detector_enabled = args.model in ["KDE", "DNNPlus", "DLA", "ICNN", "PAD"]
    attack_name = "sigmaBinary_deferred" if indicator_masking else "sigmaBinary"
    file_paths = {
        "mals": output_dir / f"{model.model_name}_mals_{attack_name}.pt",
        "advs": output_dir / f"{model.model_name}_advs_{attack_name}.pt",
    }

    # If results already exist, load them (makes reruns easier)
    if file_paths["mals"].exists() and file_paths["advs"].exists():
        log(f"[run_attack] Found existing adversarial outputs. Loading from {output_dir}")
        loaded_data = load_advs(model.model_name, attack_name, output_dir=str(output_dir))
        mals_adapt = loaded_data.get("mals")
        advs_adapt = loaded_data.get("advs")
    else:
        log("[run_attack] Starting attack generation...")
        try:
            mals_adapt, advs_adapt = attack_mal_batches(
                test_loader, model, sigmaBinary, feature_mask, device,
                detector_enabled=detector_enabled,
                oblivion=args.oblivion,
                indicator_masking=indicator_masking,
                verbose=args.verbose,
                max_iterations=args.max_iterations,
                learning_rate=args.learning_rate,
                threshold=args.threshold,
                binary_search_steps=args.binary_search_steps,
                initial_const=args.initial_const,
                confidence_bound_primary=args.confidence_bound_primary,
                confidence_bound_secondary=args.confidence_bound_secondary,
                confidence_update_interval=args.confidence_update_interval,
                sigma=args.sigma
            )
        except Exception as e:
            print(f"[ERROR] Attack generation failed: {e}", file=sys.stderr)
            raise

    # Evaluate attack
    log("[run_attack] Evaluating adversarial examples...")
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion, indicator_masking=indicator_masking,
        rounded=False, batch_size=args.batch_size, norm="L0"
    )

    # Save Adversarial Examples (final)
    save_advs(model_name=model.model_name, attack_name=attack_name,
              output_dir=str(output_dir), mals=mals_adapt, advs=advs_adapt)

    log("[run_attack] Finished. Saved adversarial examples and metadata to:", str(output_dir))


if __name__ == '__main__':
    main()

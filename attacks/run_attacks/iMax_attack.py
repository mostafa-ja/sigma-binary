"""
Reproducible runner for the Iterative Max (iMax) adversarial attack
combining multi-norm PGD (L1, L2, Linf) perturbations.

Usage example:
python -m attacks.run_attacks.iMax_attack \
  --cuda --data-path dataset/malscan_preprocessed \
  --param-path defenses/saved_parameters \
  --model DNN --step-lengths '{"L1":1.0,"L2":1.2,"Linf":0.001}' --verbose
"""

import argparse
import torch
import json
import os
import time
from pathlib import Path
import re


from utils.utils import *
from defenses.model_implementations.models import *
from attacks.baseline_attacks.iMax import iMax


def main():
    parser = argparse.ArgumentParser(description="Run iMax (multi-norm PGD) adversarial attack.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True,
                        choices=["DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"],
                        help="Model name to attack")
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative",
                        help="Indicator masking mode")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for dataset loading")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--max-iterations", type=int, default=1000, help="Maximum iterations for PGD attacks")
    parser.add_argument("--steps-max", type=int, default=5, help="Number of iterative iMax rounds")
    parser.add_argument("--binary-search-steps", type=int, default=4, help="Binary search steps for detector models")
    parser.add_argument("--initial-const", type=float, default=1.0, help="Initial balancing constant")
    parser.add_argument("--step-lengths", type=str, required=True,
                        help="Dictionary of step lengths for L1, L2, Linf norms, e.g. '{\"L1\":1.0,\"L2\":1.2,\"Linf\":0.001}'")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples",
                        help="Directory to save adversarial examples")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    

    # Parse step lengths dictionary
    raw = args.step_lengths.strip()
    # Try to make it valid JSON for PowerShell-style input
    if not raw.startswith("{"):
        raw = "{" + raw + "}"
    # Add double quotes around keys if missing
    raw = re.sub(r"([{,]\s*)([A-Za-z0-9_]+)(\s*:)", r'\1"\2"\3', raw)
    try:
        step_lengths = json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid format for --step-lengths. Example: --step-lengths '{{\"L1\":1.0,\"L2\":1.2,\"Linf\":0.001}}'\nError: {e}")


    # Prepare output directory and metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": str(device),
        "args": vars(args)
    }
    with open(output_dir / f"run_iMax_metadata_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Load dataset
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)

    # Load model
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

    attack_name = "iMax"
    file_paths = {
        "mals": output_dir / f"{model.model_name}_mals_{attack_name}.pt",
        "advs": output_dir / f"{model.model_name}_advs_{attack_name}.pt",
    }

    # Run or load
    if file_paths["mals"].exists() and file_paths["advs"].exists():
        loaded = load_advs(model.model_name, attack_name, output_dir=str(output_dir))
        mals_adapt, advs_adapt = loaded["mals"], loaded["advs"]
    else:
        mals_adapt, advs_adapt = attack_mal_batches(
            test_loader, model, iMax, feature_mask, device,
            detector_enabled=detector_enabled, oblivion=args.oblivion,
            indicator_masking=indicator_masking, verbose=args.verbose,
            max_iterations=args.max_iterations, steps_max=args.steps_max,
            step_lengths=step_lengths, binary_search_steps=args.binary_search_steps,
            initial_const=args.initial_const
        )

    # Evaluate
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion,
        indicator_masking=indicator_masking, rounded=False,
        batch_size=args.batch_size, norm="L0"
    )

    # Save
    save_advs(model_name=model.model_name, attack_name=attack_name, output_dir=str(output_dir), mals=mals_adapt, advs=advs_adapt)
    print(f"[iMax_attack] Finished for {args.model} | Saved to {output_dir}")


if __name__ == "__main__":
    main()

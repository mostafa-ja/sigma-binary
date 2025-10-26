"""
Reproducible runner for the StepwiseMax attack (mixture of one-step PGD
under L1, L2, and Linf), matching the structure of your other runners.

Example:
python -m attacks.run_attacks.StepwiseMax_attack \
  --cuda --data-path dataset/malscan_preprocessed \
  --param-path defenses/saved_parameters \
  --model DNN --batch-size 1024 \
  --max-iterations 1000 \
  --step-lengths '{"L1":1.0,"L2":0.4,"Linf":0.05}' \
  --verbose
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
import torch

from utils.utils import *
from defenses.model_implementations.models import *
# NOTE: import the attack from your Step_wise module (as in your example)
from attacks.baseline_attacks.Step_wise import StepwiseMax


def main():
    parser = argparse.ArgumentParser(description="Run StepwiseMax (mixture of PGD L1/L2/Linf one-step) attack.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True,
                        choices=["DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"])
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative",
                        help="Indicator masking mode")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (use 256 for KDE)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    parser.add_argument("--max-iterations", type=int, default=1000, help="Max StepwiseMax outer iterations")
    parser.add_argument("--binary-search-steps", type=int, default=4, help="Binary search steps (for detector models)")
    parser.add_argument("--initial-const", type=float, default=1.0, help="Initial balancing constant")
    parser.add_argument("--step-lengths", type=str, required=True,
                        help="JSON dict for step sizes, e.g. '{\"L1\":1.0,\"L2\":0.4,\"Linf\":0.05}'")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples",
                        help="Directory to save adversarial tensors")

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


    # Prepare output dir + metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": str(device),
        "args": vars(args)
    }
    with open(output_dir / f"run_StepwiseMax_metadata_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Load data
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)

    # Build model
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

    attack_name = "StepwiseMax"
    file_paths = {
        "mals": output_dir / f"{model.model_name}_mals_{attack_name}.pt",
        "advs": output_dir / f"{model.model_name}_advs_{attack_name}.pt",
    }

    # Run or load cached
    if file_paths["mals"].exists() and file_paths["advs"].exists():
        loaded = load_advs(model.model_name, attack_name, output_dir=str(output_dir))
        mals_adapt, advs_adapt = loaded["mals"], loaded["advs"]
    else:
        mals_adapt, advs_adapt = attack_mal_batches(
            test_loader, model, StepwiseMax, feature_mask, device,
            detector_enabled=detector_enabled, oblivion=args.oblivion,
            indicator_masking=indicator_masking, verbose=args.verbose,
            max_iterations=args.max_iterations, step_lengths=step_lengths,
            binary_search_steps=args.binary_search_steps, initial_const=args.initial_const
        )

    # Evaluate
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion,
        indicator_masking=indicator_masking, rounded=False,
        batch_size=args.batch_size, norm="L0"
    )

    # Save (use keyword args to match your utils implementation)
    save_advs(
        model_name=model.model_name,
        attack_name=attack_name,
        output_dir=str(output_dir),
        mals=mals_adapt,
        advs=advs_adapt,
    )

    print(f"[StepwiseMax_attack] Finished for {args.model}. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

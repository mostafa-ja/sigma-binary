import torch
import argparse
import json
from utils.utils import *
from defenses.model_implementations.models import *
from attacks.baseline_attacks.CW import *
from attacks.binary_rounding_methods.probabilistic_binary_rounding import probabilistic_binary_rounding
from attacks.binary_rounding_methods.thresholded_binary_rounding import thresholded_binary_rounding


def main():
    parser = argparse.ArgumentParser(description="Run Sigma Zero attack on malware detection models.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True, choices=[
        "DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"
    ], help="Model name to attack")
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative", 
                        help="Indicator masking mode: 'deferred' (True) or 'conservative' (False)")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack (default: adaptive attack)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for loading the dataset")
    parser.add_argument("--verbose", action="store_true", help="Print attack process details")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations for Sigma Zero attack")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate for the attack")
    parser.add_argument("--binary_search_steps_cw", type=int, default=8, help="Number of binary search steps for CW penalty")
    parser.add_argument("--binary_search_steps_penalty", type=int, default=4, help="Number of binary search steps for detector penalty")
    parser.add_argument("--initial_const_cw", type=float, default=1.0, help="Initial constant for CW penalty")
    parser.add_argument("--initial_const_penalty", type=float, default=1.0, help="Initial constant for detector penalty")
    parser.add_argument("--rounded", action="store_true", help="If set, use rounded attack results")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples", 
                        help="Directory to save adversarial example tensors")
    parser.add_argument("--rounding-function", type=str, choices=[
        "prioritized_binary_rounding", "probabilistic_binary_rounding", "thresholded_binary_rounding"
    ], default="prioritized_binary_rounding", help="Rounding function to use")
    parser.add_argument("--rounding-args", type=str, default="{}", 
                        help="Additional arguments for rounding function as a JSON string")

    args = parser.parse_args()

    # Convert the string to the actual function
    rounding_function_mapping = {
        "prioritized_binary_rounding": prioritized_binary_rounding,
        "probabilistic_binary_rounding": probabilistic_binary_rounding,
        "thresholded_binary_rounding": thresholded_binary_rounding
    }
    rounding_function = rounding_function_mapping[args.rounding_function]

    # Parse rounding arguments from JSON string
    try:
        rounding_args = json.loads(args.rounding_args)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --rounding-args. Example valid format: '{\"threshold\": 0.5}'")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load dataset
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)

    # Load the selected model
    model_mapping = {
        "DNN": lambda: MalwareDetectionDNN("DNN", device=device),
        "AT_rFGSM": lambda: MalwareDetectionDNN("AT_rFGSM", device=device),
        "AT_MaxMA": lambda: MalwareDetectionDNN("AT_MaxMA", device=device),
        "KDE": lambda: KernelDensityEstimation(MalwareDetectionDNN("DNN", device=device).float(), device=device, name="KDE"),
        "DLA": lambda: AMalwareDetectionDLA(MalwareDetectionDNN("DNN", device=device), device=device),
        "DNNPlus": lambda: AMalwareDetectionDNNPlus(MalwareDetectionDNN("DNN", device=device), device=device, model_name="DNNPlus"),
        "ICNN": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="ICNN"),
        "PAD": lambda: AdvMalwareDetectorICNN(MalwareDetectionDNN("DNN", device=device), device=device, model_name="PAD"),
    }
    model = model_mapping[args.model]()
    model.load(args.param_path)
    model.to(device)

    # Set attack parameters
    indicator_masking = args.mode == "deferred"
    detector_enabled = args.model in ["KDE", "DNNPlus", "DLA", "ICNN", "PAD"]

    # Run attack
    attack_name = "CW"
    file_paths = {
        "mals": os.path.join(f"{args.output_dir}", f"{model.model_name}_mals_{attack_name}.pt"),
        "advs": os.path.join(f"{args.output_dir}", f"{model.model_name}_advs_{attack_name}.pt"),
    }

    if os.path.exists(file_paths['mals']) and os.path.exists(file_paths['advs']):
        #print(f"Both {file_paths['mals']} and {file_paths['advs']} exist.")
        loaded_data = load_advs(model.model_name, attack_name, output_dir=args.output_dir)
        mals_adapt = loaded_data.get("mals")
        advs_adapt = loaded_data.get("advs")
    else:
        mals_adapt, advs_adapt = attack_mal_batches(
            test_loader, model, CW, feature_mask, device, 
            detector_enabled=detector_enabled, oblivion=args.oblivion,
            indicator_masking=indicator_masking, verbose=args.verbose,
            max_iterations=args.max_iterations, learning_rate=args.learning_rate, 
            binary_search_steps_cw=args.binary_search_steps_cw, binary_search_steps_penalty=args.binary_search_steps_penalty, 
            initial_const_cw=args.initial_const_cw, initial_const_penalty=args.initial_const_penalty
        )

    # Evaluate attack
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion, indicator_masking=indicator_masking, 
        rounded=args.rounded, batch_size=args.batch_size, norm="L0",
        rounding_function=rounding_function, rounding_args=rounding_args
    )

    # Determine attack name based on rounding function
    if not args.rounded:
        attack_name = f"{attack_name}"
    elif args.rounding_function == "prioritized_binary_rounding":
        attack_name = f"{attack_name}_R"
    elif args.rounding_function == "thresholded_binary_rounding":
        attack_name = f"{attack_name}_R_thresh"
    else:
        attack_name = f"{attack_name}_R_prob"

    # Save adversarial examples
    save_advs(
        model_name=model.model_name,
        attack_name=attack_name,
        output_dir=args.output_dir,
        mals=mals_adapt,
        advs=advs_adapt,
    )


if __name__ == "__main__":
    main()

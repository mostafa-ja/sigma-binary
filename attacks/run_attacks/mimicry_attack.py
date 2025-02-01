import torch
import argparse
from utils.utils import attack_mal_batches, process_ben_samples, evaluate_adversarial_examples, save_advs
from utils.utils import load_data, load_feature_mask
from defenses.model_implementations.models import *
from attacks.baseline_attacks.Mimicry import *


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attack on malware detection models.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True, choices=[
        "DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"
    ], help="Model name to attack")
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative", 
                        help="Indicator masking mode: 'deferred' (True) or 'conservative' (False)")
    parser.add_argument("--oblivion", action="store_true", help="Use oblivious attack (default: adaptive attack)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for loading the dataset")
    parser.add_argument("--verbose", action="store_true", help="Print attack process details")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples", 
                        help="Directory to save adversarial example tensors")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load dataset
    _, _, test_loader = load_data(args.data_path, args.batch_size)
    feature_mask = load_feature_mask(args.data_path, device)
    ben_samples_test = process_ben_samples(test_loader, device)

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
    detector_enabled = args.model in ["KDE", "DNNPlus","DLA", "ICNN", "PAD"]
    # Run attack
    mals_adapt, advs_adapt = attack_mal_batches(
        test_loader, model, Mimicry, feature_mask, device, 
        detector_enabled=detector_enabled, oblivion=args.oblivion,
        indicator_masking=indicator_masking, input_bens=ben_samples_test, verbose=args.verbose
    )

    # Evaluate attack
    evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion, indicator_masking=indicator_masking, 
        rounded=False, batch_size=args.batch_size, norm="L0"
    )

    # Save adversarial examples
    save_advs(
        model_name=model.model_name,
        attack_name="mimicry",
        output_dir=args.output_dir,  # Now uses custom output directory
        mals=mals_adapt,
        advs=advs_adapt,
    )


if __name__ == "__main__":
    main()

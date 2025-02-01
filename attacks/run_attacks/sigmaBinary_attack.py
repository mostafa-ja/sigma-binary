import argparse
import torch
from utils.utils import *
from defenses.model_implementations.models import *
from attacks.sigma_binary.SigmaBinary import *

def main():
    parser = argparse.ArgumentParser(description="Run sigmaBinary attack on malware detection models.")
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
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value for the attack")
    parser.add_argument("--binary-search-steps", type=int, default=4, help="Number of binary search steps for the attack")
    parser.add_argument("--initial-const", type=float, default=0.1, help="Initial constant for binary search")
    parser.add_argument("--output-dir", type=str, default="dataset/adversarial_examples", 
                        help="Directory to save adversarial example tensors")
    parser.add_argument('--confidence-bound-primary', type=float, default=0.4, help='Primary confidence bound')
    parser.add_argument('--confidence-bound-secondary', type=float, default=0.4, help='Secondary confidence bound')
    parser.add_argument('--confidence-update-interval', type=int, default=80, help='Confidence update interval')


    args = parser.parse_args()

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
    detector_enabled = args.model in ["KDE", "DNNPlus","DLA", "ICNN", "PAD"]

    # Determine attack name
    attack_name = "sigmaBinary_deferred" if indicator_masking else "sigmaBinary"
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
          test_loader, model, sigmaBinary, feature_mask, device,
          detector_enabled= detector_enabled, oblivion=args.oblivion, 
          indicator_masking=indicator_masking, verbose=args.verbose,
          max_iterations=args.max_iterations, learning_rate=args.learning_rate,
          threshold=args.threshold, binary_search_steps=args.binary_search_steps,
          initial_const=args.initial_const, confidence_bound_primary=args.confidence_bound_primary,
          confidence_bound_secondary=args.confidence_bound_secondary,
          confidence_update_interval=args.confidence_update_interval
      )
    
    # Evaluate attack
    advs_adapt, mals_adapt = evaluate_adversarial_examples(
        mal_x=mals_adapt, adv=advs_adapt, model=model, feature_mask=feature_mask,
        detector_enabled=detector_enabled, oblivion=args.oblivion, indicator_masking=indicator_masking, 
        rounded=False, batch_size=args.batch_size, norm="L0"
    )
    
    # Save Adversarial Examples
    save_advs(model_name=model.model_name, attack_name=attack_name,
              output_dir=args.output_dir, mals=mals_adapt, advs=advs_adapt)
    
if __name__ == '__main__':
    main()


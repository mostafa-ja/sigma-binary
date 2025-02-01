import torch
import argparse
from utils.utils import load_data
from defenses.model_implementations.models import *

def main():
    parser = argparse.ArgumentParser(description="Run malware detection models.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--param-path", type=str, required=True, help="Path to saved model parameters")
    parser.add_argument("--model", type=str, required=True, choices=[
        "DNN", "AT_rFGSM", "AT_MaxMA", "KDE", "DLA", "DNNPlus", "ICNN", "PAD"
    ], help="Model name to run")
    parser.add_argument("--mode", type=str, choices=["deferred", "conservative"], default="conservative", 
                        help="Prediction mode: 'deferred' (indicator_masking=True) or 'conservative' (indicator_masking=False)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for loading the dataset")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Load dataset
    _, _, test_loader = load_data(args.data_path, args.batch_size)

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
    model.to(device).double()

    # Define indicator_masking based on mode
    indicator_masking = args.mode == "deferred"

    # Run prediction
    if args.model in ["KDE", "DLA", "DNNPlus", "ICNN", "PAD"]:
        model.predict(test_loader, indicator_masking=indicator_masking)
    else:
        model.predict(test_loader)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
from typing import Any, Tuple, List
from attacks.binary_rounding_methods.prioritized_binary_rounding import *
import time

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    b_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {b_accuracy * 100:.2f}%")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    f1 = f1_score(y_true, y_pred, average='binary')
    print(f"FPR: {fpr * 100:.2f}%, FNR: {fnr * 100:.2f}%, F1 Score: {f1 * 100:.2f}%")


def load_data(root_path, batch_size):
    # Use os.path.join to construct file paths
    train_features = torch.load(os.path.join(root_path, 'train_features.pt'), weights_only=True).float()
    train_labels = torch.load(os.path.join(root_path, 'train_labels.pt'), weights_only=True).long()
    val_features = torch.load(os.path.join(root_path, 'val_features.pt'), weights_only=True).float()
    val_labels = torch.load(os.path.join(root_path, 'val_labels.pt'), weights_only=True).long()
    test_features = torch.load(os.path.join(root_path, 'test_features.pt'), weights_only=True).float()
    test_labels = torch.load(os.path.join(root_path, 'test_labels.pt'), weights_only=True).long()

    # Create datasets
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_feature_mask(root_path, device):
    
    # Construct the file path using os.path.join
    feature_mask_path = os.path.join(root_path, 'feature_mask.pt')
    
    # Load the feature mask tensor
    feature_mask = torch.load(feature_mask_path, weights_only=True, map_location=device)
    
    return feature_mask




def attack_mal_batches(
    test_loader: DataLoader,
    model: nn.Module,
    attack: Any,
    feature_mask: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    detector_enabled: bool = True,
    oblivion: bool = False,
    indicator_masking: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies adversarial attacks on malicious samples within batches and returns
    the original malicious samples and the corresponding adversarial examples.

    Args:
        test_loader (DataLoader): DataLoader for test data.
        model (nn.Module): Model to be attacked.
        attack (Any): Adversarial attack function (e.g., sigma_zero_rounded).
        feature_mask (torch.Tensor): Masking tensor indicating removable features.
        device (torch.device): Device for computation.
        detector_enabled (bool, optional): Enables/disables detector during attack.
        oblivion (bool, optional): Flag for attack configuration.
        indicator_masking (bool, optional): Enables indicator masking for suspecious samples (prob>tau).
        **kwargs: Additional parameters for the attack function (e.g., steps, lr).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Original malicious samples and adversarial examples.
    """
    # ---- prepare device + CUDA synchronization / peak mem reset ----
    use_cuda = (device.type != 'cpu') and torch.cuda.is_available()
    cuda_warmup_iters = 5  #Number of dummy CUDA operations performed before attack execution to stabilize GPU memory allocation.
    if use_cuda:
        for _ in range(cuda_warmup_iters):
            _ = torch.randn(1, device=device) * 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    # Initialization
    model.eval()
    perturbed_batches = []
    perturbed_indicator_batches = []
    mal_x_batches = []

    # Iterate through each batch in the test_loader
    for x_test, y_test in test_loader:

        # Move batch to device
        x_test, y_test = x_test.to(device), y_test.to(device)

        # Select malicious samples (label == 1)
        mal_x_batch = x_test[y_test.squeeze() == 1]

        if mal_x_batch.size(0) > 0:  # Process only if there are malicious samples
            # Generate adversarial examples for malicious samples
            perturbed_batch = attack(
                model=model,
                original_inputs=mal_x_batch,
                feature_mask=feature_mask,
                device=device,
                detector_enabled=detector_enabled,
                oblivion=oblivion,
                **kwargs
            )
            perturbed_batches.append(perturbed_batch)
            mal_x_batches.append(mal_x_batch)

            # Apply indicator masking if enabled and required
            if detector_enabled and not oblivion and indicator_masking:
                print(f"{'-'*50}")
                print('Oblivion Attack ')
                perturbed_indicator_batch = attack(
                    model=model,
                    inputs=mal_x_batch,
                    feature_mask=feature_mask,
                    device=device,
                    detector_enabled=detector_enabled,
                    oblivion=True,
                    **kwargs
                )
                perturbed_indicator_batches.append(perturbed_indicator_batch)


    # Combine all malicious samples and their perturbed counterparts
    combined_mals = torch.cat(mal_x_batches, dim=0)
    combined_advs = torch.cat(perturbed_batches, dim=0)

    # Initialize final adversarial examples based on attack conditions
    if detector_enabled and not oblivion and indicator_masking:
        combined_indicator_advs = torch.cat(perturbed_indicator_batches, dim=0)
        final_advs = combined_advs.clone()

        # Get logits and probability
        if model.model_name == 'KDE':
            logits, x_hidden = model.forward_f(final_advs)
            prob = model.forward_g(x_hidden, y_pred=0)
        else:
            logits, prob = model(final_advs)

        tau = model.tau[0].item()
        success_mask = (logits.argmax(dim=1) == 0) & (prob <= tau)

        # Update unsuccessful adversarial examples using the indicator batch
        final_advs = torch.where(success_mask.unsqueeze(-1), final_advs, combined_indicator_advs)
    else:
        final_advs = combined_advs
    
    
    
    # Finalize timing + memory
    if use_cuda:
        torch.cuda.synchronize(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = float(peak_memory_bytes) / (1024.0 ** 3)
    else:
        peak_memory_gb = None

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    metrics = {
        'elapsed_time_s': float(elapsed_time),
        'peak_memory_mb': peak_memory_gb,
    }


    print("\n")
    print(f"Evaluation on calculation cost")
    print("-----------------------------------------------")
    print(f"Execution time: {elapsed_time:.6f} seconds")
    if peak_memory_gb is not None:
        print(f"CUDA peak memory allocated: {peak_memory_gb:.3f} GB")
    else:
        print("CUDA not used; peak memory not available.")
    print("\n")

    # ---- Evaluation step ----
    _, _ = evaluate_adversarial_examples(
        mal_x=combined_mals.detach(),
        adv=final_advs.detach(),
        model=model,
        feature_mask=feature_mask.detach(),
        detector_enabled=detector_enabled,
        oblivion=oblivion,
        indicator_masking=False,
        rounded=False,
        norm='L0'
    )

    return combined_mals, final_advs


def process_ben_samples(loader, device):

    ben_x_list = []

    for x_batch, y_batch in loader:
        # Transfer data to the specified device
        x_batch = x_batch.to(torch.float32).to(device)
        y_batch = y_batch.to(device)

        # Split into malicious and benign batches
        ben_mask = y_batch.squeeze() == 0

        ben_x_list.append(x_batch[ben_mask])


    # Concatenate all batches
    ben_x = torch.cat(ben_x_list, dim=0) if ben_x_list else torch.empty(0, *x_batch.shape[1:], device=device)

    return ben_x




def calculate_distance(adv_samples: torch.Tensor, original_samples: torch.Tensor, norm: str) -> torch.Tensor:
    """
    Calculates the distance between adversarial and original samples based on the specified norm.

    Args:
        adv_samples (torch.Tensor): The adversarial samples.
        original_samples (torch.Tensor): The original samples.
        norm (str): The norm type ('L0', 'L1', 'L2').

    Returns:
        torch.Tensor: Calculated distances between each adversarial and original sample.
    """
    if norm == 'L0':
        return (adv_samples != original_samples).float().sum(dim=1)
    elif norm == 'L1':
        return torch.abs(adv_samples - original_samples).sum(dim=1)
    elif norm == 'L2':
        return torch.norm(adv_samples - original_samples, p=2, dim=1)
    else:
        raise ValueError("Unsupported norm. Use 'L0', 'L1', or 'L2'.")



def evaluate_adversarial_examples(
    mal_x: torch.Tensor,
    adv: torch.Tensor,
    model: nn.Module,
    feature_mask: torch.Tensor,
    detector_enabled: bool = False,
    oblivion: bool = False,
    indicator_masking: bool = False,
    rounded: bool = False,
    rounding_function=None,
    rounding_args=None,
    batch_size: int = 1024,
    norm: str = 'L0'
) -> torch.Tensor:
    """
    Evaluates adversarial examples against a model, calculating various distance metrics
    and success rates for different norms, optionally in batches.

    Args:
        mal_x (torch.Tensor): Original malware input.
        adv (torch.Tensor): Adversarial examples generated from mal_x.
        model (nn.Module): The trained model to be evaluated.
        feature_mask (torch.Tensor): Array used for final rounding.
        detector_enabled (bool, optional): Enables detector processing during evaluation.
        oblivion (bool, optional): Alters attack configuration if true.
        indicator_masking (bool, optional): Enables indicator masking for suspect samples.
        rounded (bool, optional): Applies final rounding on adversarial examples.
        rounding_function (callable, required if rounded=True): The function used for rounding.
        rounding_args (dict, optional): Arguments to be passed to the rounding function.
        batch_size (int, optional): Maximum samples per batch for processing.
        norm (str): Norm used for distance calculations ('L0', 'L1', 'L2').

    Returns:
        torch.Tensor: The final adversarial examples used in evaluation.

    Raises:
        ValueError: If `rounded=True` and no `rounding_function` is provided.
    """
    if rounded and rounding_function is None:
        raise ValueError("You must specify a `rounding_function` when `rounded=True`.")

    model.eval()
    torch.cuda.empty_cache()

    total_samples = mal_x.shape[0]
    num_successful_attacks = 0
    total_distances = 0
    y_pred_list = []
    mal_x_list = []
    final_adv_list = []
    all_distances: List[float] = []

    for i in range(0, total_samples, batch_size):
        mal_x_batch = mal_x[i:i + batch_size]
        adv_batch = adv[i:i + batch_size]

        # Apply rounding if specified
        if rounded:
            if rounding_function == prioritized_binary_rounding:
                final_adv_batch = rounding_function(
                    original_inputs=mal_x_batch,
                    adversarial_inputs=adv_batch,
                    model=model,
                    feature_mask=feature_mask,
                    detector_enabled=detector_enabled,
                    oblivion=oblivion
                )
            else:
                final_adv_batch = rounding_function(adv_batch, **(rounding_args or {}))
        else:
            final_adv_batch = adv_batch.clone()

        # Perform forward pass and determine successful attacks
        with torch.no_grad():
            if detector_enabled:
                if model.model_name == 'KDE':
                    logits, x_hidden = model.forward_f(final_adv_batch)
                    prob = model.forward_g(x_hidden, y_pred=0)
                else:
                    logits, prob = model(final_adv_batch)

                # Identify successful attacks
                successful_attacks_mask = (logits.argmax(dim=1) == 0) & (prob <= model.tau[0].item())

                # Exclude samples for indicator masking
                if indicator_masking:
                    exclusion_mask = (logits.argmax(dim=1) == 0) & (prob > model.tau[0].item())
                    successful_attacks_mask = successful_attacks_mask[~exclusion_mask]
                    mal_x_batch = mal_x_batch[~exclusion_mask]
                    final_adv_batch = final_adv_batch[~exclusion_mask]

            else:
                logits = model(final_adv_batch)
                successful_attacks_mask = (logits.argmax(dim=1) == 0)

        final_adv_list.append(final_adv_batch)
        y_pred_batch = (~successful_attacks_mask).long()
        y_pred_list.append(y_pred_batch)
        mal_x_list.append(mal_x_batch)

        # Calculate distances for successful attacks
        num_successful_attacks += successful_attacks_mask.sum().item()
        if successful_attacks_mask.any():
            distances = calculate_distance(final_adv_batch[successful_attacks_mask], mal_x_batch[successful_attacks_mask], norm)
            total_distances += distances.sum().item()
            all_distances.extend(distances.cpu().tolist())

    # Aggregate predictions and adversarial examples
    y_pred = torch.cat(y_pred_list)
    final_adv = torch.cat(final_adv_list)
    final_mal_x = torch.cat(mal_x_list)

    # Calculate metrics
    num_considered_malwares = len(y_pred)
    ASR = (y_pred == 0).float().mean().item() * 100
    mean_distance = total_distances / max(1, num_successful_attacks)
    median_distance = torch.median(torch.tensor(all_distances)).item() if all_distances else 0

    # Calculate IQR (Interquartile Range) and Variance
    if all_distances:
        distances_tensor = torch.tensor(all_distances)
        q1 = torch.quantile(distances_tensor, 0.25).item()
        q3 = torch.quantile(distances_tensor, 0.75).item()
        iqr = q3 - q1
        variance = torch.var(distances_tensor).item()    # IQR = Q3 - Q1, which measures the spread of the middle 50% of the data
    else:
        iqr = 0
        variance = 0

    ExS = (total_samples - num_considered_malwares) / total_samples

    asr_values = [(torch.tensor(all_distances) <= threshold).float().sum().item() / total_samples * 100 for threshold in [10,20,50,100]]

    # Print results
    print('')
    print(f"Evaluation on Adversarial Examples, Norm: {norm}, Rounded: {rounded}, rounding function:{rounding_function.__name__ if rounding_function else 'None'}")
    print("-----------------------------------------------")
    print(f"Total malwares: {total_samples} and Considered malwares : {num_considered_malwares}")
    print(f"Percentage of excluded samples is {ExS * 100:.2f}%")
    print(f"Successful attacks: {num_successful_attacks}/{num_considered_malwares}")
    print(f"Attack success rate:: {ASR:.2f}%")
    print(f"Attack success rate with k=10:: {asr_values[0]:.2f}%")
    print(f"Attack success rate with k=20:: {asr_values[1]:.2f}%")
    print(f"Attack success rate with k=50:: {asr_values[2]:.2f}%")
    print(f"Attack success rate with k=100:: {asr_values[3]:.2f}%")

    print(f"Mean distance (successful): {mean_distance:.2f}")
    print(f"Median distance (successful): {median_distance:.2f}")
    print(f"IQR (successful): {iqr:.2f}")
    print(f"Variance (successful): {variance:.2f}")

    '''
    # Update the loop for top-N distance metrics
    top_n_values = [1000, 2000, 3000]
    for top_n in top_n_values:
        if num_successful_attacks >= top_n:
            median_top_n_distance = median_of_top_n_distances(final_adv, final_mal_x, y_pred, top_n, norm=norm)
            print(f"Median distance (TOP {top_n} successful): {median_top_n_distance:.2f}")
    '''

    return final_adv, final_mal_x






def save_advs(
    model_name,
    attack_name,
    output_dir="adversarial_examples",
    **tensors,
):
    """
    Save model data tensors as .pt files.

    Args:
        model_name (str): Name of the model (e.g., "ICNN").
        attack_name (str): Name of the attack (e.g., "FGSM").
        output_dir (str): Directory to save the .pt files.
        **tensors: Keyword arguments for tensors (e.g., mals, advs, mals_detect, etc.).
                   Provide tensors dynamically depending on the configuration.

    Required Tensors:
        - If `detector_enabled` is False:
            * mals
            * advs
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Required tensor keys
    required_keys = ["mals", "advs"]

    # Check that all required tensors are provided
    missing_keys = [key for key in required_keys if key not in tensors]
    if missing_keys:
        raise ValueError(f"Missing required tensors: {missing_keys}")

    # Generate file paths and tensors to save
    file_prefix = model_name
    tensors_to_save = {
        f"{file_prefix}_mals_{attack_name}.pt": tensors["mals"],
        f"{file_prefix}_advs_{attack_name}.pt": tensors["advs"],
    }

    # Save tensors as .pt files
    for file_name, tensor in tensors_to_save.items():
        file_path = os.path.join(output_dir, file_name)
        torch.save(tensor.clone(), file_path)
        print(f"File saved in: {file_path}")
    print('')

    




def load_advs(model_name, attack_name, output_dir="adversarial_examples"):
    """
    Load the saved .pt tensors.

    Args:
        model_name (str): Name of the model (e.g., "ICNN").
        output_dir (str): Directory where the .pt files are stored.

    Returns:
        dict: A dictionary containing loaded tensors.
    """
    file_prefix = model_name
    file_paths = {
        "mals": os.path.join(output_dir, f"{file_prefix}_mals_{attack_name}.pt"),
        "advs": os.path.join(output_dir, f"{file_prefix}_advs_{attack_name}.pt"),
    }

    loaded_tensors = {}
    for key, file_path in file_paths.items():
        if os.path.exists(file_path):
            loaded_tensors[key] = torch.load(file_path, weights_only=True)
        else:
            print(f"Warning: {file_path} not found.")

    return loaded_tensors



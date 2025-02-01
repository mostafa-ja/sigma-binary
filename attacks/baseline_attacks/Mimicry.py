
import torch



def Mimicry(model, original_inputs, input_bens, feature_mask, detector_enabled=False, oblivion=False, device='cpu', verbose=True):
    """
    Performs mimicry-based adversarial attack to find benign samples similar to malicious ones
    and modifies malicious samples to deceive the model.

    Args:
        model: The PyTorch model under attack.
        original_inputs: Malicious inputs (batch of examples).
        input_bens: Benign inputs for mimicry.
        removal_array: Mask specifying which features to modify.
        detector_enabled: Whether a detector is enabled in the model.
        oblivion: Specific condition for detector logic.
        device: Device to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
        final_adv: Tensor of adversarial examples.
    """
    torch.cuda.empty_cache()

    # Move necessary tensors to the specified device
    model = model.to(device)
    model.eval()
    original_inputs = original_inputs.to(device)
    input_bens = input_bens.to(device)
    feature_mask = feature_mask.to(device)
    fixed_features_mask = torch.bitwise_and(~feature_mask.unsqueeze(0), original_inputs.to(torch.uint8))

    # Initialize storage tensors and counters
    final_adv = original_inputs.clone()
    total_samples = 0
    valid_sample_count = 0
    valid_distances_sum = 0.0

    # Iterate through the malicious inputs
    for idx, mal_sample in enumerate(original_inputs):

        # Combine malicious sample with benign samples
        ben_x_batch_new = torch.cat((mal_sample.unsqueeze(0) , input_bens), dim=0)
        inputs_combined = torch.bitwise_or(fixed_features_mask[idx].unsqueeze(0), ben_x_batch_new.to(torch.uint8)).float()
        # Pass the combined inputs through the model
        with torch.no_grad():
            if detector_enabled:
                if model.model_name == 'KDE':
                    logits, x_hidden = model.forward_f(inputs_combined)
                    prob = model.forward_g(x_hidden, y_pred=0)
                else:
                    logits, prob = model.forward(inputs_combined)
            else:
                logits = model.forward(inputs_combined)

            success_mask = (logits.argmax(dim=1) == 0) & (prob <= model.tau[0].item() if detector_enabled and not oblivion else True)

        # Calculate L1 distances and filter successful inputs
        distances = (mal_sample != inputs_combined).sum(dim=1)
        valid_distances = distances[success_mask]
        valid_inputs = inputs_combined[success_mask]

        # Update counters and tensors with valid results
        total_samples += 1
        if valid_distances.numel() > 0:
            min_distance, min_index = torch.min(valid_distances, dim=0)
            final_adv[idx] = valid_inputs[min_index]
            valid_sample_count += 1
            valid_distances_sum += min_distance.item()

        # Logging progress
        if verbose and ((idx + 1) % 512 == 0):
            current_mean_distance = valid_distances_sum / valid_sample_count if valid_sample_count > 0 else 0.0
            print(f"Processed {idx + 1:4}/{original_inputs.size(0):4} samples | "
                  f"Success: {valid_sample_count:4}/{total_samples:4} | "
                  f"Avg. Dist: {current_mean_distance:3.2f}")

    return final_adv

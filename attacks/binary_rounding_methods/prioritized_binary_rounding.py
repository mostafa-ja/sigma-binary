import torch


def prioritized_binary_rounding(original_inputs, adversarial_inputs, model, feature_mask, detector_enabled=False,
                                oblivion=False, max_considered_features=300, device='cpu',stable=True,random_seed=924):
    """
    Apply rounding adjustments to adversarial examples by prioritizing changes based on model's predictions.

    Parameters:
    - original_inputs (Tensor): The original input tensor.
    - adversarial_inputs (Tensor): The adversarially perturbed input tensor.
    - model (nn.Module): The model used for evaluation of the input samples.
    - feature_mask (Tensor): A tensor indicating which features should remain unchanged.
    - detector_enabled (bool): Flag indicating whether a detector is enabled (optional).
    - oblivion (bool): Flag indicating whether the attacker is unaware of a detector (optional).
    - stable (bool): If True, uses stable sort (preserves equal-value order). 
                     If False, breaks ties randomly but deterministically if seed is given.
    - random_seed (int or None): Seed for reproducible random tie-breaking when stable=False.

    Returns:
    - final_inputs (Tensor): The adjusted adversarial examples after rounding.
    """

    # Ensure all tensors are on the same device
    device = original_inputs.device
    original_inputs, adversarial_inputs, feature_mask = original_inputs.to(device), adversarial_inputs.to(device), feature_mask.to(device)

    # Optional reproducibility for random tie-breaking
    if (not stable) and (random_seed is not None):
        torch.manual_seed(random_seed)
    
    # Create mask for features that can be modified (1 means modifiable, 0 means fixed)
    modifiable_mask = torch.bitwise_or(feature_mask.expand(original_inputs.shape[0], -1), 1 - original_inputs.to(torch.uint8)).bool()

    # Set the model to evaluation mode
    model.eval()

    # Compute model predictions and prepare for change prioritization
    with torch.no_grad():

        # Model inference with or without detector
        if detector_enabled :
            if model.model_name == 'KDE':
                logits, hidden_layer = model.forward_f(original_inputs)
                if not oblivion:
                    probabilities = model.forward_g(hidden_layer, y_pred=0)
                else :
                    probabilities = None
            else:
                logits, probabilities = model.forward(original_inputs)
        else:
            logits, probabilities = model.forward(original_inputs), None


        # If detector is enabled and we don't ignore it, create a mask for inactive samples
        if detector_enabled and not oblivion:
            threshold = model.tau[0].item()
            inactive_mask = (logits.argmax(dim=1) == 0) & (probabilities <= threshold)
        else:
            inactive_mask = (logits.argmax(dim=1) == 0)

        # Active samples are those that are not inactive
        active_samples_mask = ~inactive_mask

        # Calculate absolute changes between adversarial and original inputs
        change_magnitude = (adversarial_inputs - original_inputs).abs()

        # Create a mask for significant changes (greater than a threshold)
        significant_change_mask = change_magnitude > 0.05
        
        # Random tie-breaking if stable=False
        if not stable:
            eps = 1e-6 * torch.rand_like(change_magnitude)
            change_magnitude = change_magnitude + eps
        
        sorted_change_indices = change_magnitude.argsort(dim=1, descending=True, stable=stable)

        # Determine the changes to be applied based on adversarial inputs
        proposed_changes = torch.where(adversarial_inputs > original_inputs, 1,
                                       torch.where(adversarial_inputs < original_inputs, 0, original_inputs)).float().to(device)

    # Clone original inputs to apply final adjustments
    final_inputs = original_inputs.clone()

    # Set a reasonable upper bound on the number of significant changes per sample
    max_changes_per_sample = min(significant_change_mask.sum(dim=1).max().item(), max_considered_features)

    # Apply changes iteratively, ensuring that only non-fixed features are modified
    for i in range(max_changes_per_sample):
        if not active_samples_mask.any():
            break

        active_sample_indices = torch.where(active_samples_mask)[0]

        # Update final input tensor based on the sorted indices and modifiable mask
        indices_to_change = sorted_change_indices[active_sample_indices, i]
        final_inputs[active_sample_indices, indices_to_change] = torch.where(
            modifiable_mask[active_sample_indices, indices_to_change],
            proposed_changes[active_sample_indices, indices_to_change],
            final_inputs[active_sample_indices, indices_to_change]
        )

        # Update the active mask by re-evaluating the model predictions after applying changes
        with torch.no_grad():

            if detector_enabled :
                if model.model_name == 'KDE':
                    logits, hidden_layer = model.forward_f(final_inputs[active_sample_indices])
                    if not oblivion:
                        probabilities = model.forward_g(hidden_layer, y_pred=0)
                    else :
                        probabilities = None
                else:
                    logits, probabilities = model.forward(final_inputs[active_sample_indices])
            else:
                logits, probabilities = model.forward(final_inputs[active_sample_indices]), None


            # Recompute the inactive mask if detector is enabled and update the active mask accordingly
            if detector_enabled and not oblivion:
                inactive_mask = (logits.argmax(dim=1) == 0) & (probabilities <= threshold)
            else:
                inactive_mask = (logits.argmax(dim=1) == 0)

            # Update the active sample mask to reflect which samples remain active
            active_samples_mask[active_sample_indices] = (~inactive_mask)

    return final_inputs.clamp(0, 1)
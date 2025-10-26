from utils.attack_utils import forward_model, update_bounds_and_consts
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def BGA(
    model: nn.Module,
    original_inputs: Tensor,
    feature_mask: Tensor,
    max_iterations: int = 1000,
    verbose: bool = False,
    detector_enabled: bool = False,
    oblivion: bool = False,
    binary_search_steps: int = 4,
    initial_const = 1.0,
    device: torch.device = torch.device('cpu'),
) -> (Tensor):
    """
    BGA idea from Al-Dujaili et al. (2018)

    This function performs a discrete, feature-based adversarial attack
    on a binary input space (e.g., tabular or presence/absence features).
    It attempts to modify as few features as possible to cause model misclassification.

    Parameters:
    -----------
    model : nn.Module
        The trained model to be attacked. Must return logits (and optionally probabilities).

    original_inputs : Tensor
        The original input tensor of shape (batch_size, num_features).
        Each element is typically binary (0 or 1).

    feature_mask : Tensor
        A binary tensor of shape (num_features,) or (batch_size, num_features),
        indicating which features are fixed (0) and which can be modified (1).

    max_iterations : int, optional (default=5000)
        Maximum number of inner optimization steps for gradient-based feature selection.

    verbose : bool, optional (default=False)
        If True, prints detailed information about the optimization progress and results.

    detector_enabled : bool, optional (default=False)
        Enables the auxiliary detector (if available in the model) for adversarial detection-aware attacks.

    oblivion : bool, optional (default=False)
        If True, ignores the detector during optimization (useful for ablation studies).

    binary_search_steps : int, optional (default=6)
        Number of outer binary search iterations for balancing the CONST parameter
        when detector is enabled.

    initial_const : float, optional (default=1.0)
        The initial constant used to scale the detector loss term.

    device : torch.device, optional (default='cpu')
        Device on which to perform computations ('cpu' or 'cuda').

    Returns:
    --------
    output : Tensor
        The generated adversarial examples (binary tensor with same shape as original_inputs).
    """

    # ---- Initialization ----
    model.eval()
    original_inputs = original_inputs.to(device)
    feature_mask = feature_mask.to(device)
    batch_size, num_features = original_inputs.shape
    targets = torch.ones(batch_size, dtype=torch.long, device=device)

    # BGA specific initialization
    sqrt_m = torch.sqrt(torch.tensor(num_features, dtype=torch.float32, device=device))

    # Loss criterion
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Mask for features that can be modified (cannot remove original features)
    modifiable_features_mask = torch.bitwise_or(
        feature_mask.expand(batch_size, -1), 1 - original_inputs.to(torch.uint8)
    )
    
    # ---- MODIFICATION START: Check for initially successful samples ----
    with torch.no_grad():
        initial_logits, initial_prob = forward_model(original_inputs, model, detector_enabled, oblivion)
        # Determine the detector threshold (tau) if applicable
        is_active = detector_enabled and not oblivion
        tau = model.tau[0].item() if is_active else 0.0
        
        # Define the success condition
        initially_successful_mask = (initial_logits.argmax(dim=1) == 0)
        if is_active:
            initially_successful_mask &= (initial_prob <= tau)

    # Tracking variables
    best_delta = torch.zeros_like(original_inputs, device=device)
    o_success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Binary search variables
    CONST = torch.full((batch_size,), initial_const, dtype=torch.float32, device=device)
    lower_bound = torch.zeros(batch_size, dtype=torch.float32, device=device)
    upper_bound = torch.full((batch_size,), 1e8, dtype=torch.float32, device=device)

    # Outer loop for binary search on CONST
    outer_iters = binary_search_steps if (detector_enabled and not oblivion) else 1
    for outer_step in range(outer_iters):
        if verbose and detector_enabled and not oblivion:
            print(f"{'-'*50}")
            print(f"Outer step: {outer_step + 1}, CONST: {torch.unique(CONST)}")

        # Initialize perturbation for this outer step
        delta = torch.zeros_like(original_inputs, requires_grad=True, device=device)
        success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Inner optimization loop
        for step in range(max_iterations):
            adv_inputs = (original_inputs + delta).clamp(0, 1) # Ensure valid inputs
            logits, prob = forward_model(adv_inputs, model, detector_enabled, oblivion)

            # Compute loss
            is_active = detector_enabled and not oblivion
            tau = model.tau[0].item() if is_active else 0.0

            loss = criterion(logits, targets)
            if is_active:
                loss = CONST * loss + (tau - prob)


            # Update success masks and store the best delta found so far
            with torch.no_grad():
                prime_success = (logits.argmax(dim=1) == 0)
                success_mask = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)
                success_samples |= success_mask
                o_success_samples |= success_mask
                best_delta = torch.where(success_mask.unsqueeze(-1), delta.data.clone(), best_delta)

            # Logging
            if verbose and (step == 0 or ((step + 1) % max(1, (max_iterations // 10)) == 0)):
                dist = (adv_inputs != original_inputs).float().sum(dim=1)
                avg_dist = dist[o_success_samples].float().mean().item() if o_success_samples.any() else float('nan')
                print(
                    f"Iteration {step + 1:5} | "
                    f"Current Success: {success_mask.sum().item():4}/{success_samples.sum().item():4} | "
                    f"Loop Success = {success_samples.sum().item():4}/{batch_size} | "
                    f"Total Success: {o_success_samples.sum().item():4}/{batch_size} | "
                    f"Best Dist (avg) = {avg_dist:3.2f} | "
                    f"Loss = {loss.mean().item():5.2f}"
                )
                #print('/n')
            
            # Early stop if all samples have succeeded
            if o_success_samples.all():
                if verbose:
                    print(f"All samples succeeded at inner step {step+1}; breaking inner loop.")
                break


            # 1. Compute gradient (we want to ascend the loss)
            grad_vars = torch.autograd.grad(loss.mean(), delta)
            gradients = grad_vars[0].data

            grad4insertion = (gradients >= 0) * modifiable_features_mask * gradients
            grad4removal = (gradients < 0) * modifiable_features_mask * gradients
            gradients = grad4removal + grad4insertion

            # 2. Compute the L2 norm of the gradient for each sample (the threshold)
            grad_norm = torch.norm(gradients, p=2, dim=1, keepdim=True)

            # 3. Create the sign-correction term
            sign_correction = 1. - 2. * adv_inputs

            # 4. Apply the BGA rule to find which features to update
            x_update = (sqrt_m * sign_correction * gradients >= grad_norm).float()

            # 5. Apply the updates using bitwise logic
            with torch.no_grad():
                # XOR flips the bits according to the update mask
                adv_inputs_flipped = adv_inputs.bool() ^ x_update.bool()
                # OR with original inputs enforces the "add-only" constraint
                next_adv_inputs = (adv_inputs_flipped | original_inputs.bool()).float()
                # Update delta for the next iteration
                delta.data = next_adv_inputs - original_inputs

            

        # Update bounds and constants for binary search
        CONST, upper_bound, lower_bound = update_bounds_and_consts(
            outer_step, binary_search_steps, CONST, upper_bound, lower_bound, prime_success, update_factor=100.
        )

    # Final adversarial examples are the original inputs plus the best-found perturbations
    output = (original_inputs + best_delta).clamp(0, 1)
    
    # ---- MODIFICATION: Restore the initially successful samples at the very end ----
    output[initially_successful_mask] = original_inputs[initially_successful_mask]

    return output
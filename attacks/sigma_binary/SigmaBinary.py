from utils.attack_utils import forward_model, update_bounds_and_consts, update_confidence
from attacks.binary_rounding_methods.prioritized_binary_rounding import prioritized_binary_rounding

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn


# success_mask: success samples in inner loop
# success_mask_R: success rounded samples in inner loop
# success_samples: total success samples in inner loop
# success_samples_R: total success rounded samples in inner loop
# o_success_samples: total success samples
# o_success_samples_R: total success rounded samples


def sigmaBinary(
    model: nn.Module,
    original_inputs: Tensor,
    feature_mask: Tensor,
    max_iterations: int = 1000,
    learning_rate: float = 0.5,
    sigma: float = 1e-4,
    threshold: float = 0.3,
    t: float = 0.01,
    verbose: bool = False,
    grad_norm: float = torch.inf,
    detector_enabled: bool = False,
    oblivion: bool = False,
    binary_search_steps: int = 5,
    initial_const = 1.0,
    initial_primary_confidence: float = 1e-4,
    initial_secondary_confidence: float = 1e-4,
    device: torch.device = torch.device('cpu'),
    confidence_bound_primary: float = 1.,
    confidence_bound_secondary: float = 1.,
    confidence_update_interval: int = 50,
    stable: bool = True
) -> Tensor:
    """
    Optimizes adversarial examples with L0 constraints using iterative methods.

    Parameters:
    - model (nn.Module): The model to evaluate adversarial examples.
    - original_inputs (Tensor): The original input samples.
    - feature_mask (Tensor): Mask indicating features that should remain fixed.
    - max_iterations (int): The maximum number of iterations for optimization.
    - learning_rate (float): The learning rate for the optimizer.
    - sigma (float): A small constant to control the L0 approximation.
    - threshold (float): A threshold for making zero those features with small perturbation.
    - t (float): The step size for updating the threshold.
    - verbose (bool): Whether to print detailed progress information.
    - grad_norm (float): The gradient normalization parameter.
    - detector_enabled (bool): Whether to use the detector model for evaluation.
    - oblivion (bool): Flag to determine if the oblivion mode should be applied.
    - binary_search_steps (int): The number of binary search steps for adjusting the confidence bounds.
    - initial_const (float): The initial value for the constant used in the loss function.
    - initial_primary_confidence (float): Initial confidence for the primary loss term.
    - initial_secondary_confidence (float): Initial confidence for the secondary loss term.
    - device (str): The device ('cpu' or 'cuda') where the tensors are located.
    - primary_confidence_bound (float): The bound for the primary confidence term.
    - secondary_confidence_bound (float): The bound for the secondary confidence term.
    - confidence_update_interval (int): The interval at which to update the confidence values.

    Returns:
    - final_adv (Tensor): The optimized adversarial examples.
    """
    
    # Initialization
    model.eval()
    original_inputs = original_inputs.to(device)
    batch_size, num_features = original_inputs.shape

    # Create mask for non-fixed features
    modifiable_features_mask = torch.bitwise_or(
        feature_mask.expand(batch_size, -1), 1 - original_inputs.to(torch.uint8)
    ).bool()

    # Initialize tracking variables
    best_dist = torch.full((batch_size,), num_features, device=device, dtype=torch.float32)
    best_delta = torch.zeros_like(original_inputs, device=device)
    o_success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)
    o_success_samples_R = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Initialize CONST and bounds
    CONST = torch.full((batch_size,), initial_const, dtype=torch.float32, device=device)
    lower_bound = torch.zeros(batch_size, dtype=torch.float32, device=device)
    upper_bound = torch.full((batch_size,), 1e8, dtype=torch.float32, device=device)

    # Outer loop for binary search
    outer_iters = binary_search_steps if (detector_enabled and not oblivion) else 1
    for outer_step in range(outer_iters):
        if verbose and detector_enabled and not oblivion:
            print(f"{'-'*50}")
            print(f"Outer step: {outer_step + 1}, CONST: {torch.unique(CONST)}")

        # Initialize perturbation delta
        delta = torch.zeros_like(original_inputs, requires_grad=True, device=device)
        delta.register_hook(lambda grad: grad * modifiable_features_mask)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=learning_rate / 100.0)

        # Initialize dynamic variables
        threshold_matrix = torch.full_like(original_inputs, threshold, device=device)
        primary_confidence = torch.full((batch_size,), initial_primary_confidence, device=device)
        secondary_confidence = torch.full((batch_size,), initial_secondary_confidence, device=device)
        prime_success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)
        success_samples_R = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stagnation_counter = torch.zeros(batch_size, dtype=torch.int32, device=device)
        best_loss_inloop = torch.full((batch_size,), float('inf'), device=device, dtype=torch.float32)

        o_stagnated_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Inner optimization loop
        for step in range(max_iterations):
            adv_inputs = original_inputs + delta
            logits, prob = forward_model(adv_inputs, model, detector_enabled, oblivion)

            # Compute losses
            ben_score, mal_score = logits[:, 0], logits[:, 1]
            tau = model.tau[0].item() if detector_enabled and not oblivion else 0


            # Update success masks and distances
            with torch.no_grad():
                prime_success |= (logits.argmax(dim=1) == 0)
                success_mask = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)
                success_samples |= success_mask
                o_success_samples |= success_mask

                if step == 0 or (step + 1) % 10 == 0:
                    adv_inputs_rounded = prioritized_binary_rounding(original_inputs, adv_inputs, model, feature_mask, detector_enabled, oblivion,stable=stable)
                    rounded_logits, rounded_prob = forward_model(adv_inputs_rounded, model, detector_enabled, oblivion)

                    success_mask_R = (rounded_logits.argmax(dim=1) == 0) & ((rounded_prob <= tau) if detector_enabled and not oblivion else True)
                    success_samples_R |= success_mask_R
                    o_success_samples_R |= success_mask_R

                    dist = (adv_inputs_rounded != original_inputs).float().sum(dim=1)
                    improved_mask = success_mask_R & (dist < best_dist)
                    best_dist = torch.where(improved_mask, dist, best_dist)
                    best_delta = torch.where(improved_mask.unsqueeze(-1), delta.data.clone(), best_delta)


                if (step + 1) % confidence_update_interval == 0:
                  if o_stagnated_mask.any():
                      primary_confidence = update_confidence(primary_confidence, o_stagnated_mask & success_mask, o_stagnated_mask & ~success_mask, confidence_bound_primary)
                      if detector_enabled and not oblivion:
                          secondary_confidence = update_confidence(secondary_confidence, o_stagnated_mask & success_mask, o_stagnated_mask & ~success_mask, confidence_bound_secondary)

            
            if detector_enabled and not oblivion:
                primary_loss = CONST * torch.clamp(mal_score - ben_score + primary_confidence, min=0.0) + torch.clamp(prob - tau + secondary_confidence, min=0.0)
            else:
                primary_loss = torch.clamp(mal_score - ben_score + primary_confidence, min=0.0)

            # Combined loss and optimization
            l0_approx = delta.square() / (delta.square() + sigma)
            l0_normalized = l0_approx.sum(dim=1) / num_features
            total_loss = (primary_loss + l0_normalized)
            total_loss_mean = total_loss.mean()

            with torch.no_grad():
                # Track the best loss and stagnation counter
                improved_mask_inloop = success_mask & (total_loss < best_loss_inloop)
                best_loss_inloop = torch.where(improved_mask_inloop, total_loss, best_loss_inloop)

                stagnation_counter = torch.where(improved_mask_inloop, 0, stagnation_counter + 1)
                o_stagnated_mask |= ((stagnation_counter >= 1000) & success_samples)




            if verbose and (step == 0 or ((step + 1) % (max_iterations // 5) == 0)):
                avg_best_dist = best_dist[o_success_samples_R].float().mean().item() if o_success_samples_R.any() else float('nan')
                print(
                    f"Iteration {step + 1:5} | "
                    f"LR = {scheduler.get_last_lr()[0]:.2f} | "
                    f"Current NotRound/Round = {success_mask.sum().item():4}/{success_mask_R.sum().item():4} | "
                    f"Loop NotRounded/Rounded = {success_samples.sum().item():4}/{success_samples_R.sum().item():4} | "
                    f"Total NotRound/Round = {o_success_samples.sum().item():4}/{o_success_samples_R.sum().item():4} | "
                    f"Best Dist (avg) = {avg_best_dist:3.2f} | "
                    f"Loss = {total_loss_mean.item():5.2f} | "
                )

            optimizer.zero_grad()
            total_loss_mean.backward()
            delta.grad /= delta.grad.norm(p=grad_norm, dim=1, keepdim=True).clamp_min(1e-12)
            optimizer.step()
            scheduler.step()

            # Enforce bounds and threshold
            with torch.no_grad():
                delta.data.add_(original_inputs).clamp_(0, 1).sub_(original_inputs)

                threshold_matrix[~success_mask] -= t * scheduler.get_last_lr()[0]
                threshold_matrix[success_mask] += t * scheduler.get_last_lr()[0]
                threshold_matrix.clamp_(0, 1)

                delta.data[(delta.data.abs() < threshold_matrix)] = 0

        # Update bounds and constants
        CONST, upper_bound, lower_bound = update_bounds_and_consts(
            outer_step, binary_search_steps, CONST, upper_bound, lower_bound, prime_success
        )

    # Final outputs
    output =  prioritized_binary_rounding(original_inputs, (original_inputs + best_delta).detach(), model, feature_mask, detector_enabled, oblivion,stable=stable)
    output =  prioritized_binary_rounding(original_inputs, output.detach(), model, feature_mask, detector_enabled, oblivion,stable=stable)

    return output

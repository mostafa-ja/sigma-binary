from utils.attack_utils import forward_model, update_bounds_and_consts
import torch
from torch import Tensor, nn


def torch_arctanh(x, eps=1e-6):
    """Apply arctanh transformation with numerical stability."""
    return torch.atanh((2 * x - 1) * (1 - eps))

def CW(
    model: nn.Module,
    original_inputs: Tensor,
    feature_mask: Tensor,
    max_iterations: int = 1000,
    learning_rate: float = 0.5,
    sigma: float = 1e-4,
    verbose: bool = True,
    detector_enabled: bool = True,
    oblivion: bool = False,
    binary_search_steps_cw: int = 1,
    binary_search_steps_penalty: int = 1,
    initial_const_cw: float = 1.,
    initial_const_penalty: float = 1.,
    primary_confidence: float = 1e-4,
    secondary_confidence: float = 1e-4,
    device: str = 'cpu'
):

    # Prepare model and input tensors
    model.eval()
    inputs = original_inputs.to(device)
    batch_size, num_features = inputs.shape
    inputs_tanh = torch_arctanh(inputs.clone().detach())

    # Mask for non-fixed features
    non_fixed_features_mask = torch.bitwise_or(
        feature_mask.expand(batch_size, -1), 1 - inputs.to(torch.uint8)
    ).bool()

    # Track best perturbations
    best_l2 = torch.full((batch_size,), num_features, device=device)
    best_delta = torch.zeros_like(inputs, device=device)
    o_o_success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)
    o_bestadv = inputs.clone().detach()

    # Initialize CONST and bounds
    CONST_cw = torch.full((batch_size,), initial_const_cw, dtype=torch.float32, device=device)
    lower_bound_cw = torch.zeros(batch_size, dtype=torch.float32, device=device)
    upper_bound_cw = torch.full((batch_size,), 1e8, dtype=torch.float32, device=device)

    for outer_step_cw in range(binary_search_steps_cw):
        if verbose :
            print(f"{'-'*50}")
            print(f"Outer step(CW): {outer_step_cw + 1}, CONST(CW): {torch.unique(CONST_cw)}")

        o_success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize CONST and bounds
        CONST = torch.full((batch_size,), initial_const_penalty, dtype=torch.float32, device=device)
        lower_bound = torch.zeros(batch_size, dtype=torch.float32, device=device)
        upper_bound = torch.full((batch_size,), 1e8, dtype=torch.float32, device=device)

        # Define outer loop iterations based on detector state and oblivion setting
        outer_iters = binary_search_steps_penalty if (detector_enabled and not oblivion) else 1

        for outer_step in range(outer_iters):
            if verbose and detector_enabled and not oblivion:
                print(f"{'-'*50}")
                print(f"Outer step: {outer_step + 1}, CONST: {torch.unique(CONST)}")

            active_inputs_tanh = torch_arctanh(inputs.clone().detach()).to(device)

            # Initialize delta for adversarial optimization
            delta = torch.zeros_like(active_inputs_tanh, requires_grad=True, device=device)
            delta.register_hook(lambda grad: grad * non_fixed_features_mask)

            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam([delta], lr=learning_rate)

            success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_iterations):
                # Generate adversarial examples and compute model logits
                adv_inputs_tanh = active_inputs_tanh + delta
                adv_inputs = (torch.tanh(adv_inputs_tanh) ) * 0.5 + 0.5

                logits, prob = forward_model(adv_inputs, model, detector_enabled, oblivion)

                # Compute losses
                ben_score, mal_score = logits[:, 0], logits[:, 1]
                tau = model.tau[0].item() if detector_enabled and not oblivion else 0

                # Success mask and update best solutions
                with torch.no_grad():
                    success_mask = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)
                    success_samples |= success_mask
                    o_success_samples |= success_mask
                    o_o_success_samples |= success_mask

                    #l2dist = torch.sum((adv_inputs -  inputs) ** 2, dim=1)
                    l2dist = torch.norm(adv_inputs - inputs, p=2, dim=1)
                    improved_mask = success_mask & (l2dist < best_l2)
                    best_l2= torch.where(improved_mask, l2dist, best_l2)
                    best_delta = torch.where(improved_mask.unsqueeze(-1), delta.data.clone(), best_delta)
                    o_bestadv = torch.where(improved_mask.unsqueeze(-1), adv_inputs.clone(), o_bestadv)


                # Combined loss
                dl_loss = torch.clamp(mal_score - ben_score + primary_confidence, min=0.0)
                if detector_enabled and not oblivion:
                    dl_loss += CONST * torch.clamp(prob - tau + secondary_confidence, min=0.0)

                # Combined loss for decision and sparsity
                adv_loss = (CONST_cw * dl_loss + l2dist).mean()

                # Verbose output for debugging
                if verbose and (step == 0 or ((step + 1) % (max_iterations // 10) == 0)):
                    avg_best_dist = best_l2[o_o_success_samples].float().mean().item() if o_o_success_samples.any() else float('nan')
                    print(
                        f"Iteration {step + 1:5} | LR = {learning_rate:.2f} | "
                        f"Current Success: {success_mask.sum().item():4}/{success_samples.sum().item():4} | "
                        f"Loop Success = {success_samples.sum().item():4}/{batch_size} | "
                        f"Total Success: {o_o_success_samples.sum().item():4}/{batch_size} | "
                        f"Best Distance (avg): {avg_best_dist:6.2f} | "
                        f"Loss: {adv_loss.item():7.2f}"
                    )

                optimizer.zero_grad()
                adv_loss.backward()
                optimizer.step()


            # Update bounds and constants
            CONST, upper_bound, lower_bound = update_bounds_and_consts(
                outer_step, binary_search_steps_penalty, CONST, upper_bound, lower_bound, success_samples,update_factor = 100.
            )

        # Update bounds and constants
        CONST_cw, upper_bound_cw, lower_bound_cw = update_bounds_and_consts(
            outer_step_cw, binary_search_steps_cw, CONST_cw, upper_bound_cw, lower_bound_cw, o_success_samples,update_factor = 10.
        )


    return o_bestadv

from utils.attack_utils import forward_model, update_bounds_and_consts
import torch
from torch import Tensor, nn


def PGD(
    model: nn.Module,
    original_inputs: Tensor,
    feature_mask: Tensor,
    max_iterations: int = 1000,
    step_length: float = 0.5,
    norm: str = 'l2',
    verbose: bool = False,
    detector_enabled: bool = False,
    oblivion: bool = False,
    binary_search_steps: int = 4,
    initial_const = 1.0,
    device: str = 'cpu',
) -> Tensor:


    # Initialization
    model.eval()
    original_inputs = original_inputs.to(device)
    batch_size, num_features = original_inputs.shape

    # Create mask for non-fixed features
    modifiable_features_mask = torch.bitwise_or(
        feature_mask.expand(batch_size, -1), 1 - original_inputs.to(torch.uint8)
    )

    # Initialize tracking variables
    best_delta = torch.zeros_like(original_inputs, device=device)
    o_success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

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

        # Initialize dynamic variables
        success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Inner optimization loop
        for step in range(max_iterations):
            adv_inputs = original_inputs + delta
            logits, prob = forward_model(adv_inputs, model, detector_enabled, oblivion)
            
            # Compute losses
            ben_score, mal_score = logits[:, 0], logits[:, 1]
            tau = model.tau[0].item() if detector_enabled and not oblivion else 0


            # Update success masks and distances
            with torch.no_grad():
                success_mask = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)
                success_samples |= success_mask
                o_success_samples |= success_mask

                dist = (adv_inputs != original_inputs).float().sum(dim=1)
                best_delta = torch.where(success_mask.unsqueeze(-1), delta.data.clone(), best_delta)

            primary_loss = (mal_score - ben_score)
            if detector_enabled and not oblivion:
                primary_loss += CONST * (prob - tau)


            if verbose and (step == 0 or ((step + 1) % (max_iterations // 10) == 0)):
                avg_dist = dist[o_success_samples].float().mean().item() if o_success_samples.any() else float('nan')
                print(
                    f"Iteration {step + 1:5} | "
                    f"LR = {step_length:.2f} | "
                    f"Current Success: {success_mask.sum().item():4}/{success_samples.sum().item():4} | "
                    f"Loop Success = {success_samples.sum().item():4}/{batch_size} | "
                    f"Total Success: {o_success_samples.sum().item():4}/{batch_size} | "
                    f"Best Dist (avg) = {avg_dist:3.2f} | "
                    f"Loss = {primary_loss.mean().item():5.2f} | "
                )

            # Compute gradient
            grad_vars = torch.autograd.grad(primary_loss.mean(), delta)
            gradients = -(grad_vars[0].data)
            #print('torch.abs(gradients).sum() : ',torch.abs(gradients).sum(dim=-1))

            grad4insertion = (gradients >= 0) *(adv_inputs < 1.) * gradients
            grad4removal = (gradients < 0) * (adv_inputs > 0.) * modifiable_features_mask * gradients

            gradients = grad4removal + grad4insertion

            # Norm
            if norm == 'linf':
                perturbation = torch.sign(gradients).float()

            elif norm == 'l2':
                #l2norm = gradients.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                #perturbation = (gradients / l2norm )
                l2norm = torch.linalg.norm(gradients, dim=-1, keepdim=True)
                perturbation = (gradients / (l2norm + 1e-20)).float()

            elif norm == 'l0':
                # consider just features of a sample which are not updated yet(because our update is 0to1 or 1to0 not stepwise)
                un_mod = torch.abs(delta.data) <= 1e-6
                gradients = gradients * un_mod
                max_grad = torch.topk(torch.abs(gradients).view(gradients.size(0), -1), 1, dim=-1)[0]
                #print('max_grad ',max_grad)
                perturbation = (torch.abs(gradients) >= max_grad).float() * torch.sign(gradients).float()

                if torch.all(success_mask):
                    break
                perturbation[success_mask] = 0.

            else:
                raise ValueError("Expect 'l0' or 'l2' or 'linf' norm.")

            with torch.no_grad():
                # Update x_next
                delta.data = (delta.data + perturbation * step_length)

                delta.data.add_(original_inputs).clamp_(0, 1).sub_(original_inputs)


        # Update bounds and constants
        CONST, upper_bound, lower_bound = update_bounds_and_consts(
            outer_step, binary_search_steps, CONST, upper_bound, lower_bound, success_samples, update_factor = 100.
        )

    return (original_inputs + best_delta)

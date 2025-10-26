from utils.attack_utils import forward_model, update_bounds_and_consts
from attacks.binary_rounding_methods.thresholded_binary_rounding import thresholded_binary_rounding
import torch
from torch import Tensor, nn



def PGD_one_step(
    model: nn.Module,
    original_inputs: Tensor,
    feature_mask: Tensor,
    max_iterations: int = 1,
    step_length: float = 0.5,
    norm: str = 'L2',
    detector_enabled: bool = False,
    oblivion: bool = False,
    CONST = 1.0,
    device: torch.device = torch.device('cpu')
) -> Tensor:

    # loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Initialization
    model.eval()
    original_inputs = original_inputs.to(device)
    feature_mask = feature_mask.to(device) # Move feature_mask to the specified device
    batch_size, num_features = original_inputs.shape
    targets = torch.ones(batch_size, dtype=torch.long, device=device)


    # Create mask for non-fixed features
    modifiable_features_mask = torch.bitwise_or(
        feature_mask.expand(batch_size, -1), 1 - original_inputs.to(torch.uint8)
    )


    # Initialize perturbation delta
    delta = torch.zeros_like(original_inputs, requires_grad=True, device=device)

    # Initialize dynamic variables
    success_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Inner optimization loop
    for step in range(max_iterations):
        adv_inputs = original_inputs + delta

        # Compute loss
        logits, prob = forward_model(adv_inputs, model, detector_enabled, oblivion)

        is_active = detector_enabled and not oblivion
        tau = model.tau[0].item() if is_active else 0.0

        loss = criterion(logits, targets)
        if is_active:
            loss = CONST * loss + (tau - prob)

        # Compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), delta)
        gradients = (grad_vars[0].data)

        grad4insertion = (gradients >= 0) *(adv_inputs < 1.) * gradients
        grad4removal = (gradients < 0) * (adv_inputs > 0.) * modifiable_features_mask * gradients

        gradients = grad4removal + grad4insertion

        # Norm
        if norm == 'Linf':
            perturbation = torch.sign(gradients).float()

        elif norm == 'L2':
            l2norm = torch.linalg.norm(gradients, dim=-1, keepdim=True)
            perturbation = (gradients / (l2norm + 1e-20)).float()

        elif norm == 'L1':
            # consider just features of a sample which are not updated yet(because our update is 0to1 or 1to0 not stepwise)
            un_mod = torch.abs(delta.data) <= 1e-6
            gradients = gradients * un_mod
            max_grad = torch.topk(torch.abs(gradients).view(gradients.size(0), -1), 1, dim=-1)[0]
            #print('max_grad ',max_grad)
            perturbation = (torch.abs(gradients) >= max_grad).float() * torch.sign(gradients).float()

        else:
            raise ValueError("Expect 'L1' or 'L2' or 'Linf' norm.")

        with torch.no_grad():
            # Update x_next
            delta.data = (delta.data + perturbation * step_length)

            delta.data.add_(original_inputs).clamp_(0, 1).sub_(original_inputs)



    return (original_inputs + delta.data)


def StepwiseMax(
    original_inputs,
    model, feature_mask,
    attack_list=["L1", "L2", "Linf"],
    step_lengths={"L1": 1.0, "L2": 0.05, "Linf": 0.002},
    max_iterations=100,
    detector_enabled=False, oblivion=False, binary_search_steps=6, initial_const=1.0, verbose=False, device=torch.device('cpu')):

    """
      Stepwise max attack (mixture of pgd-L1, pgd-l2, pgd-linf).

      Args:
          x: Input data tensor (shape: [batch_size, feature_dim])
          model: Victim model
          attack_list: List of attack norms (default: ["L1", "l2", "linf"])
          step_lengths: Dictionary mapping norm to its step length (default: {"L1": 1.0, "l2": 0.05, "linf": 0.002})
          max_iterations: Maximum number of iterations (default: 100)

      Returns:
          Adversarial examples tensor (same shape as x)
    """

    # loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Initialize tracking variables
    model.eval()
    batch_size, num_features = original_inputs.shape
    best_adv = original_inputs.clone().detach().to(device)
    
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


        n, red_n = original_inputs.shape
        adv_x = original_inputs.detach().clone()
        pert_x_cont = None
        prev_done = None
        for step in range(max_iterations):

            with torch.no_grad():

                # Compute loss
                logits, prob = forward_model(thresholded_binary_rounding(adv_x), model, detector_enabled, oblivion)

                is_active = detector_enabled and not oblivion
                tau = model.tau[0].item() if is_active else 0.0
                done = (logits.argmax(dim=1) == 0) & ((prob <= tau) if is_active else True)


            if torch.all(done):
                break
            if step == 0:
                adv_x[~done] = original_inputs[~done]  # recompute the perturbation under other penalty factors
                prev_done = done
            else:
                adv_x[~done] = pert_x_cont[~done[~prev_done]]
                prev_done = done

            if verbose and (step == 0 or ((step + 1) % (max_iterations // 10) == 0)):
                print(
                      f"Iteration {step + 1:5} | "
                      f"Attack effectiveness {done.sum().item() / done.size()[0] * 100:.2f}%."
                  )

            num_sample_red = torch.sum(~done).item()
            pertbx = []
            for norm in attack_list:
                step_length = step_lengths.get(norm, 1.)

                perturbation = PGD_one_step(model, adv_x[~done], feature_mask, max_iterations=1,
                                  step_length=step_length, norm=norm,
                                  detector_enabled=detector_enabled, oblivion=oblivion, CONST=CONST[~done], device=device)

                pertbx.append(perturbation)

            with torch.no_grad():
                pertbx = torch.vstack(pertbx)
                n_attacks = len(attack_list)

                # Compute loss
                logits, prob = forward_model(thresholded_binary_rounding(pertbx), model, detector_enabled, oblivion)
                is_active = detector_enabled and not oblivion
                tau = model.tau[0].item() if is_active else 0.0

                loss = criterion(logits, torch.ones(logits.shape[0], dtype=torch.long, device=device))
                if is_active:
                    loss = CONST[~done].repeat(n_attacks) * loss + (tau - prob)

                _done = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)


                max_v = loss.amax() if loss.amax() > 0 else 0.
                loss[_done] += max_v

                pertbx = pertbx.reshape(n_attacks, num_sample_red, red_n).permute([1, 0, 2])
                loss = loss.reshape(n_attacks, num_sample_red).permute(1, 0)
                _2, s_idx = loss.max(dim=-1)
                pert_x_cont = pertbx[torch.arange(num_sample_red), s_idx]
                adv_x[~done] = thresholded_binary_rounding(pert_x_cont)


        with torch.no_grad():

            logits, prob = forward_model(thresholded_binary_rounding(adv_x), model, detector_enabled, oblivion)
            tau = model.tau[0].item() if detector_enabled and not oblivion else 0

            prime_success = (logits.argmax(dim=1) == 0)
            done = (logits.argmax(dim=1) == 0) & ((prob <= tau) if detector_enabled and not oblivion else True)
            best_adv = torch.where(done.unsqueeze(-1), thresholded_binary_rounding(adv_x).clone().detach(), best_adv)


            if verbose:
                print(f"step-wise max: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")

        # Update bounds and constants
        CONST, upper_bound, lower_bound = update_bounds_and_consts(
              outer_step, binary_search_steps, CONST, upper_bound, lower_bound, prime_success, update_factor = 100.)


    output = thresholded_binary_rounding(best_adv)
    
    # ---- MODIFICATION: Restore the initially successful samples at the very end ----
    output[initially_successful_mask] = original_inputs[initially_successful_mask]

    return output
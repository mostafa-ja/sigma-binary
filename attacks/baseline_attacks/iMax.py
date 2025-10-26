from attacks.baseline_attacks.PGD import *
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def iMax(original_inputs, model, feature_mask, attack_list=["L1", "L2", "Linf"], step_lengths={"L1": 1.0, "L2": 0.05, "Linf": 0.005}, max_iterations=1000, steps_max=5,
             detector_enabled=False, oblivion=False, binary_search_steps=4, initial_const=1.0, verbose=False, device=torch.device('cpu')):
    """
    PGD_Max adversarial attack using the PGD function, modified to keep
    samples that are initially successful unperturbed.
    """

    # loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    batch_size = original_inputs.shape[0]
    targets = torch.ones(batch_size, dtype=torch.long, device=device)

    model.eval()

    # 1. Identify Initially Successful Samples
    with torch.no_grad():
        # Compute loss on original inputs
        logits, prob = forward_model(original_inputs, model, detector_enabled, oblivion)

        is_active = detector_enabled and not oblivion
        tau = model.tau[0].item() if is_active else 0.0

        # Define success condition (same as later in the code)
        initially_successful_mask = (logits.argmax(dim=1) == 0) & ((prob <= tau) if is_active else True)

        loss = criterion(logits, targets)
        if is_active:
            # Use a placeholder for initial loss comparison, doesn't matter much for successful samples
            loss = loss + (tau - prob)

    pre_loss = loss
    n = original_inputs.shape[0]
    adv_x = original_inputs.detach().clone().to(original_inputs.device)
    
    # 2. Modify stop_flag to immediately stop the attack on successful samples
    stop_flag = initially_successful_mask.clone() # Initially successful samples are marked as 'done'
    

    for t in range(steps_max):
        # Only run on samples where stop_flag is False
        num_remaining = (~stop_flag).sum().item()
        if num_remaining <= 0:
            break

        pertbx = []

        for norm in attack_list:
            # PGD is only run on the remaining samples (~stop_flag)
            perturbation = PGD(model, adv_x[~stop_flag], feature_mask, max_iterations=max_iterations,
                                step_length=step_lengths[norm], norm=norm,
                                detector_enabled=detector_enabled, oblivion=oblivion, binary_search_steps=binary_search_steps,
                                initial_const=initial_const, verbose=verbose, device=device)

            pertbx.append(perturbation)

        # pertbx.shape = a list of (number of attacks ,(num_remaining ,features))
        pertbx = torch.vstack(pertbx)
        # here pertbx.shape = a tensor (num_remaining *number of attacks samples, features)

        with torch.no_grad():
            # Compute loss
            logits, prob = forward_model(pertbx, model, detector_enabled, oblivion)

            is_active = detector_enabled and not oblivion
            tau = model.tau[0].item() if is_active else 0.0

            done = (logits.argmax(dim=1) == 0) & ((prob <= tau) if is_active else True)

            loss = criterion(logits, torch.ones(len(attack_list) * num_remaining, dtype=torch.long, device=device)) # Expand targets here
            if is_active:
                loss = loss + (tau - prob)

            # for a sample, if there is at least one successful attack, we will select the one with maximum loss;
            # while if no attacks evade the victim successful, all perturbed examples are reminded for selection
            max_v = loss.amax()
            loss[done] += max_v

            loss = loss.reshape(len(attack_list), num_remaining ).permute(1, 0) #(num_remaining ,number of attacks)
            done = done.reshape(len(attack_list), num_remaining ).permute(1, 0) #(num_remaining ,number of attacks)

            success_flag = torch.any(done, dim=-1) #(num_remaining )

            pertbx = pertbx.reshape(len(attack_list), num_remaining , original_inputs.shape[1]).permute([1, 0, 2])#(num_remaining ,attacks,features)
            _, indices = loss.max(dim=-1) # ans:(samples), max loss among attacks which worked, and max loss among all attacks for sample , none of them worked
            
            # Update only for the remaining samples
            adv_x[~stop_flag] = pertbx[torch.arange(num_remaining ), indices]
            a_loss = loss[torch.arange(num_remaining ), indices]
            pre_stop_flag = stop_flag.clone()
            
            # Update stop_flag for the next iteration
            stop_flag[~stop_flag] = (torch.abs(pre_loss[~stop_flag] - a_loss) < 1e-16) | success_flag
            pre_loss[~pre_stop_flag] = a_loss


    output = adv_x.clamp_(0, 1)

    return output
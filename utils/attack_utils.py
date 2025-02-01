import torch
from torch import Tensor, nn



def forward_model(inputs: Tensor, model: nn.Module, detector_enabled: bool, oblivion: bool):
    """
    Perform the forward pass for the given model.
    """
    if detector_enabled:
        if model.model_name == 'KDE':
            logits, x_hidden = model.forward_f(inputs)
            prob = model.forward_g(x_hidden, y_pred=0)
        else:
            logits, prob = model.forward(inputs)
    else:
        logits, prob = model.forward(inputs), None
    return logits, prob

def update_confidence(confidence: Tensor, mask_success: Tensor, mask_failure: Tensor, bound: float) -> Tensor:
    """
    Adjust confidence values for successful and failed samples.
    """
    confidence[mask_success] = -bound
    confidence[mask_failure] = bound
    return confidence


# Helper Function: Update Bounds and Constants
def update_bounds_and_consts(
    outer_step: int,
    binary_search_steps: int,
    CONST: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_bound: torch.Tensor,
    success_samples_R: torch.Tensor,
    update_factor = 100.
):
    """
    Update bounds and constants for binary search.
    """
    if (outer_step + 1) == binary_search_steps:
        return CONST, upper_bound, lower_bound

    # Update bounds
    upper_bound[success_samples_R] = torch.minimum(upper_bound[success_samples_R], CONST[success_samples_R])
    lower_bound[~success_samples_R] = torch.maximum(lower_bound[~success_samples_R], CONST[~success_samples_R])

    # Update CONST values
    CONST[success_samples_R] /= update_factor
    CONST[~success_samples_R] *= update_factor

    # Adjust bounds dynamically
    updated_upper_limit_mask = upper_bound < 1e8
    updated_lower_limit_mask = lower_bound > 0

    CONST[success_samples_R & updated_lower_limit_mask] = (
        (lower_bound[success_samples_R & updated_lower_limit_mask] +
         upper_bound[success_samples_R & updated_lower_limit_mask]) / 2.0
    )
    CONST[~success_samples_R & updated_upper_limit_mask] = (
        (lower_bound[~success_samples_R & updated_upper_limit_mask] +
         upper_bound[~success_samples_R & updated_upper_limit_mask]) / 2.0
    )

    return CONST, upper_bound, lower_bound
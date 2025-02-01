import torch

def thresholded_binary_rounding(tensor, threshold=0.5):
    """
    Binarizes a tensor based on a given threshold.

    Args:
        tensor (torch.Tensor): Input tensor with values between 0 and 1.
        threshold (float): Threshold value. Values above this become 1, others become 0.

    Returns:
        torch.Tensor: Binary tensor with 0s and 1s.

    Raises:
        ValueError: If the tensor contains values outside the range [0, 1].
    """
    if not torch.all((0 <= tensor) & (tensor <= 1)):
        raise ValueError("Input tensor must have values between 0 and 1.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    return (tensor > threshold).float()

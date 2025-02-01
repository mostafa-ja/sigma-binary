import torch

def probabilistic_binary_rounding(tensor, seed=42):
    """
    Binarizes a tensor based on the probability represented by each element.

    Each element is treated as the probability of being 1, and random sampling
    is used to decide the binary value.

    Args:
        tensor (torch.Tensor): Input tensor with values between 0 and 1,
                               representing probabilities.
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.

    Returns:
        torch.Tensor: Binary tensor with 0s and 1s.

    Raises:
        ValueError: If the tensor contains values outside the range [0, 1].
    """
    if not torch.all((0 <= tensor) & (tensor <= 1)):
        raise ValueError("Input tensor must have values between 0 and 1.")

    torch.manual_seed(seed)  # Set the random seed for reproducibility

    # Generate random numbers of the same shape as the input tensor
    random_tensor = torch.rand_like(tensor)
    # Compare the tensor with the random numbers to determine binary values
    return (tensor > random_tensor).float()
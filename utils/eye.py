import numpy as np
import torch


def eye(*args, toTorch=True):
    """
    returns an identity matrix with given dimensions
    Last dimension is always the size of the identity matrix
    eye(5, 6) -> 5, 6x6 Identity matrices, out dim = [5, 6, 6] 
    eye(4, 5, 6) -> tensor of 4, 5, 6x6 Identity matrices, out dim = [4, 5, 6, 6]

    Continuing from the last example, the implementation creates a 6x6 identity matrix and tiles i
    """
    assert_vals = [isinstance(_, int) for _ in args]
    assert all(
        assert_vals
    ), "To construct a multi-dimensional Identity matrix, each dimension must be an integer."

    eye = np.eye(args[-1])
    multidim_eye = np.tile(
        eye, (*args[:-1], 1, 1))  # (4, 5, 1, 1) following docstring example
    if toTorch:
        multidim_eye = torch.tensor(multidim_eye, dtype=torch.float32)
    return multidim_eye

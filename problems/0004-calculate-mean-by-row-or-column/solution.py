import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    a_t = torch.as_tensor(matrix, dtype=torch.float)

    if mode == 'column':
        t = 0
    elif mode == 'row':
        t = 1
    else:
        raise ValueError("Invalid mode")

    return a_t.mean(dim=t)
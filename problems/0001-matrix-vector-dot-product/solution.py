import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    # 先检查维度
    if a_t.ndim != 2 or b_t.ndim != 1:
        return torch.tensor(-1)

    m, n = a_t.shape

    if n != b_t.shape[0]:
        return torch.tensor(-1)

    result = torch.zeros(m)

    for i in range(m):
        for j in range(n):
            result[i] += a_t[i, j] * b_t[j]

    return result
import torch


def gather(v: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    v = torch.gather(input=v, dim=-1, index=index)
    return v.reshape(-1, 1, 1, 1)

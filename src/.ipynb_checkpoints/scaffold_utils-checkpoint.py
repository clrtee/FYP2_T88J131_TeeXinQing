from __future__ import annotations
from typing import List
import numpy as np
import torch


def get_parameters_numpy(model) -> List[np.ndarray]:
    """Model state_dict -> list of numpy arrays (Flower format)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_numpy(model, parameters: List[np.ndarray]) -> None:
    """List of numpy arrays -> load into model state_dict."""
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


def params_to_vec(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into one 1D vector."""
    return torch.cat([p.contiguous().view(-1) for p in params])


def vec_to_params(vec: torch.Tensor, like_params: List[torch.Tensor]) -> List[torch.Tensor]:
    """Unflatten vector back into list of tensors with shapes like like_params."""
    out = []
    offset = 0
    for p in like_params:
        n = p.numel()
        out.append(vec[offset: offset + n].view_as(p))
        offset += n
    return out


def model_params_as_list(model) -> List[torch.Tensor]:
    """Return model parameters as a list (in a fixed order)."""
    return [p for p in model.parameters()]


def zeros_like_params(model) -> List[torch.Tensor]:
    """Create zero tensors shaped like model parameters."""
    return [torch.zeros_like(p.data) for p in model.parameters()]

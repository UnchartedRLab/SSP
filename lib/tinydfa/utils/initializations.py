import torch

from enum import Enum
from functools import partial


def uniform(d_grad, max_d_feedback, device=None):
    return torch.rand((d_grad, max_d_feedback), device=device) * 2 - 1


def gaussian(d_grad, max_d_feedback, device=None):
    return torch.randn((d_grad, max_d_feedback), device=device)


def orthogonal(d_grad, max_d_feedback, device=None):
    random_matrix = torch.zeros((d_grad, max_d_feedback), device=device)
    torch.nn.init.orthogonal_(random_matrix)
    return random_matrix


class RandomMatrixInitialization(Enum):
    UNIFORM = partial(uniform)
    GAUSSIAN = partial(gaussian)
    ORTHOGONAL = partial(orthogonal)

    def __call__(self, *args):
        return self.value(*args)

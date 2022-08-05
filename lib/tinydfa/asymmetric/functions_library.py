from enum import Enum
from functools import partial

from .functions import (
    identity_forward,
    identity_backward,
    tanh_forward,
    tanh_backward,
    relu_forward,
    relu_backward,
    gelu_forward,
)


class ForwardFunction(Enum):
    IDENTITY = partial(identity_forward)
    TANH = partial(tanh_forward)
    RELU = partial(relu_forward)
    GELU = partial(gelu_forward)

    def __call__(self, *args):
        return self.value(*args)


class BackwardFunction(Enum):
    IDENTITY = partial(identity_backward)
    TANH = partial(tanh_backward)
    RELU = partial(relu_backward)

    def __call__(self, *args):
        return self.value(*args)

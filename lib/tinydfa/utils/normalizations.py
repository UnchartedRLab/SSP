import numpy as np

from enum import Enum
from functools import partial


def no_normalization(feedback, dfa_manager=None, d_grad=None, d_feedback=None):
    return feedback


def fan_in(feedback, dfa_manager=None, d_grad=None, d_feedback=None):
    return feedback / np.sqrt(d_grad)


def fan_out(feedback, dfa_manager=None, d_grad=None, d_feedback=None):
    return feedback / np.sqrt(d_feedback)


class FeedbackNormalization(Enum):
    NO_NORMALIZATION = partial(no_normalization)
    FAN_IN = partial(fan_in)
    FAN_OUT = partial(fan_out)

    def __call__(self, *args):
        return self.value(*args)

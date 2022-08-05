import torch.nn.functional as F


def gelu_forward(input):
    return F.gelu(input)


# TODO: implement gelu_backward!

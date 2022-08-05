import torch


def tanh_forward(input):
    return torch.tanh(input)


def tanh_backward(input, grad_output):
    return grad_output * (1 - torch.pow(torch.tanh(input), 2))

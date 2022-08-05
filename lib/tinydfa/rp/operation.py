import torch.nn as nn

from abc import ABC, abstractmethod


class RandomProjectionOperation(ABC, nn.Module):
    # Virtual class for RP implementation.
    # TODO: switch to protocol?
    @abstractmethod
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.d_grad, self.max_d_feedback, self.grad_device = None, None, None
        super(RandomProjectionOperation, self).__init__()

    @abstractmethod
    def initialize(self, d_grad, max_d_feedback, grad_device):
        self.d_grad = d_grad
        self.max_d_feedback = max_d_feedback
        self.grad_device = grad_device

    @abstractmethod
    def forward(self, gradient):
        raise NotImplementedError(
            "RandomProjectionOperation class must implement a forward returning a RP of the provided gradient!"
        )

    @staticmethod
    def print_tensor_statistics(tensor, name):
        zero_count = (tensor == 0).float().mean().item() * 100
        median_pos_values = tensor.abs()[tensor.abs() > 0].median() if len(tensor.abs()[tensor.abs() > 0]) != 0 else -1
        print(
            f"{name} -- norm:{tensor.norm():.3f}, non-zero med.: {median_pos_values:.4f}, abs. mean:{tensor.abs().mean():.4f}, "
            f"0s: {zero_count:.0f}%, min/max: {tensor.min():.4f}/{tensor.max():.4f}."
        )
        # print(tensor)

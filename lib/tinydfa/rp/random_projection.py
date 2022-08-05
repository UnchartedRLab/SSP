import torch

from .operation import RandomProjectionOperation
from ..utils.initializations import RandomMatrixInitialization


class RandomProjection(RandomProjectionOperation):
    def __init__(self, pre_initialization_shape=None, initialization=RandomMatrixInitialization.UNIFORM, verbose=False):
        super(RandomProjection, self).__init__(verbose=verbose)

        self.initialization = (
            initialization.value if type(initialization) == RandomMatrixInitialization else initialization
        )
        self.pre_initialized = False

        if pre_initialization_shape is not None:
            # Pre-initialization can be used to allocate the matrix directly at initialization:
            if len(pre_initialization_shape) != 2:
                raise ValueError(
                    f"Invalid shape for pre_initialization_shape {pre_initialization_shape}! Should be of "
                    f"dimension 2: (d_grad, max_d_feedback)."
                )

            self.register_buffer("feedback_matrix", self.initialization(*pre_initialization_shape))
            self.pre_initialized = True

    def initialize(self, d_grad, max_d_feedback, grad_device):
        super(RandomProjection, self).initialize(d_grad, max_d_feedback, grad_device)

        if not self.pre_initialized:
            # The feedback matrix has not yet been created, create it from scratch:
            self.register_buffer("feedback_matrix", self.initialization(self.d_grad, self.max_d_feedback, grad_device))
        else:
            # We already have a pre-initialized feedback matrix, cut it into the right shape:
            if self.d_grad > self.feedback_matrix.shape[0] or self.max_d_feedback > self.feedback_matrix.shape[1]:
                raise ValueError(
                    f"Pre-initialized feedback matrix of insufficient shape "
                    f"{self.feedback_matrix.shape}, "
                    f"({self.d_grad, self.max_d_feedback}) required!"
                )

            self.feedback_matrix = self.feedback_matrix[: self.d_grad, : self.max_d_feedback].to(self.grad_device)

    def forward(self, gradient):
        rp = torch.mm(gradient, self.feedback_matrix)

        if self.verbose:
            RandomProjection.print_tensor_statistics(gradient, "gradient")
            RandomProjection.print_tensor_statistics(rp, "rp")
        return rp

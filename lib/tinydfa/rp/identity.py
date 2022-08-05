from .operation import RandomProjectionOperation


class IdentityProjection(RandomProjectionOperation):
    def __init__(self, verbose=False):
        super(IdentityProjection, self).__init__(verbose=verbose)

    def initialize(self, d_grad, max_d_feedback, grad_device):
        super(IdentityProjection, self).initialize(d_grad, max_d_feedback, grad_device)

        if d_grad != max_d_feedback:
            raise ValueError(
                f"Different d_grad ({d_grad}) and max_d_feedback ({max_d_feedback}) are not supported for "
                f"identity projection!"
            )

    def forward(self, gradient):
        return gradient

import numpy as np
import torch

from ..random_projection import RandomProjection
from ...utils.initializations import RandomMatrixInitialization


class RandomProjectionDP(RandomProjection):
    def __init__(
        self,
        sigma_privacy=0.01,
        tau_feedback_privacy=1,
        pre_initialization_shape=None,
        initialization=RandomMatrixInitialization.UNIFORM,
        verbose=False,
    ):
        super().__init__(
            pre_initialization_shape=pre_initialization_shape, initialization=initialization, verbose=verbose
        )

        self.sigma_privacy = sigma_privacy
        self.tau_feedback_privacy = tau_feedback_privacy

    def forward(self, gradient):
        rp = super(RandomProjectionDP, self).forward(gradient)

        if self.tau_feedback_privacy is not None:
            tau_feedback_clip = (self.tau_feedback_privacy / (rp.norm(2, dim=1))).unsqueeze(1).repeat(1, rp.shape[1])
            rp[rp >= 1.0] = 1.0
            rp = rp * tau_feedback_clip

        noise = torch.randn(rp.shape, device=rp.device) / np.sqrt(self.max_d_feedback) * self.sigma_privacy * 10

        # print(f"NOISE -- norm: {noise.norm(2)}, abs. mean: {noise.abs().mean()}, max: {noise.abs().max()}")
        # print(f"RP -- norm: {rp.norm(2)}, abs. mean: {rp.abs().mean()}, max: {rp.abs().max()}")

        rp += noise

        return rp

import torch

from .random_projection import RandomProjection
from ..utils.initializations import RandomMatrixInitialization


class TernarizedRandomProjection(RandomProjection):
    def __init__(
        self,
        ternarization_treshold=0.1,
        pre_initialization_shape=None,
        initialization=RandomMatrixInitialization.UNIFORM,
        verbose=False,
    ):
        super(TernarizedRandomProjection, self).__init__(pre_initialization_shape, initialization, verbose)
        self.ternarization_treshold = ternarization_treshold

        self.cosine_similarity = torch.nn.CosineSimilarity() if self.verbose else None

    def forward(self, gradient):
        batch_size = gradient.shape[0]

        gradient_pos = (gradient > (self.ternarization_treshold / batch_size)).float()
        gradient_neg = (gradient < (-self.ternarization_treshold / batch_size)).float()

        ternarized_gradient = gradient_pos - gradient_neg

        rp = torch.mm(ternarized_gradient, self.feedback_matrix) / 130

        rp_ref = torch.mm(gradient, self.feedback_matrix)

        if self.verbose:
            angle = self.cosine_similarity(rp, rp_ref)
            print(f"\n RP match: mean. {float(angle.mean())}, min. {float(angle.min())}, max. {float(angle.max())}")
            TernarizedRandomProjection.print_tensor_statistics(gradient, "gradient")
            TernarizedRandomProjection.print_tensor_statistics(ternarized_gradient, "ternarized_gradient")
            TernarizedRandomProjection.print_tensor_statistics(rp, "rp")

        return rp

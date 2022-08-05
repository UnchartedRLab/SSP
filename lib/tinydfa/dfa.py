import numpy as np
import torch
import torch.nn as nn

from .rp import RandomProjection
from .utils.dimensions import remove_indices
from .utils.normalizations import FeedbackNormalization


class DFABackend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dfa_manager):
        ctx.dfa_manager = dfa_manager  # Access to global DFA information in the backward.
        return input

    @staticmethod
    def backward(ctx, grad_output):
        dfa_manager = ctx.dfa_manager

        # If training with DFA feedbacks, perform the random projection and send it to the feedback points:
        if not dfa_manager.no_feedbacks:
            d_grad = np.prod(remove_indices(grad_output.shape, dfa_manager.batch_dimensions))  # gradient dimension
            if dfa_manager.d_output != d_grad:
                raise RuntimeError(
                    f"Mismatch between output dimension {dfa_manager.d_output} " f"and gradient dimension {d_grad}!"
                )

            random_projection = ctx.dfa_manager.rp(grad_output.reshape(-1, d_grad))

            # Go through the feedback points and backward with the RP on each of them:
            for layer in dfa_manager.feedback_layers:
                if layer.record_feedback_point:
                    feedback_point = layer.feedback_point
                    feedback_shape = feedback_point.shape

                    # TODO: in distributed, manually moving the RP like that might be compromising performance.
                    d_feedback = np.prod(
                        remove_indices(feedback_shape, layer.batch_dimensions)
                    )  # layer feedback dimension
                    feedback = random_projection[:, :d_feedback].view(*feedback_shape).to(feedback_point.device)

                    feedback = dfa_manager.normalization(feedback, dfa_manager, d_grad, d_feedback)

                    feedback_point.backward(feedback)
                    layer.feedback_point = None  # TODO: not applied in shallow/no_feedbacks mode.

            del random_projection, feedback

        return grad_output, None  # Gradients for output (for top layer) and dfa_manager (None).


class DFAManager(nn.Module):
    def __init__(
        self,
        feedback_layers,  # TODO: add autodetection
        rp_operation=RandomProjection(),
        normalization=FeedbackNormalization.FAN_OUT,
        no_feedbacks=False,
        batch_dimensions=(0,),  # TODO: non-sequential batch dims
    ):
        # TODO: the status of "shallow", especially regarding no_feedbacks, should be clarified.
        # TODO: see how to support shallow training in distributed setup.
        # TODO: check there are no stray FeedbackLayer through an option.
        super(DFAManager, self).__init__()
        self.feedback_layers = nn.ModuleList(feedback_layers)
        self.rp = rp_operation
        self.normalization = normalization.value if type(normalization) == FeedbackNormalization else normalization
        self.no_feedbacks = no_feedbacks
        self.batch_dimensions = batch_dimensions

        # Set the batch dims of all DFALayers with no specific batch dims:
        for dfa_layer in self.feedback_layers:
            if dfa_layer.batch_dimensions is None:
                dfa_layer.batch_dimensions = self.batch_dimensions

        self.dfa = DFABackend.apply  # Custom DFA autograd function that actually handles the backward.

        # Random feedback matrix and its dimensions:
        self.max_d_feedback = 0
        self.d_output = None

        self.initialized = False  # Initialization is ran after first pass, when all shapes can be known.

        self._allow_bp_all = False
        self._record_feedback_point_all = True
        self._use_bp = False

    def forward(self, output):
        if not (self.initialized or self.no_feedbacks):
            # If we are training, but aren't initialized:
            # - Get the size of the output (d_output);
            # - Get the size of the largest feedback (max_d_feedback);
            # - Initialize the RP operation with (d_output * max_d_feedback).
            # Wrap this process in utils
            self.d_output = int(np.prod(remove_indices(output.shape, self.batch_dimensions)))

            # Find the largest feedback:
            for layer in self.feedback_layers:
                d_feedback = int(np.prod(remove_indices(layer.feedback_point.shape, layer.batch_dimensions)))
                if d_feedback > self.max_d_feedback:
                    self.max_d_feedback = d_feedback

            self.rp.initialize(self.d_output, self.max_d_feedback, output.device)

            self.initialized = True

        return self.dfa(output, self)

    @property
    def allow_bp_all(self):
        return self._allow_bp_all

    @allow_bp_all.setter
    def allow_bp_all(self, value):
        for layer in self.feedback_layers:
            layer.allow_bp = value

        self._allow_bp_all = value

    @property
    def record_feedback_point_all(self):
        return self._record_feedback_point_all

    @record_feedback_point_all.setter
    def record_feedback_point_all(self, value):
        for layer in self.feedback_layers:
            layer.record_feedback_point = value

        self._record_feedback_point_all = value

    @property
    def use_bp(self):
        return self._use_bp

    @use_bp.setter
    def use_bp(self, value):
        self.allow_bp_all = value
        self.no_feedbacks = value
        self.record_feedback_point_all = not value

        self._use_bp = value


class FeedbackLayer(nn.Module):
    def __init__(self, name=None, batch_dimensions=None, allow_bp=False, record_feedback_point=True):
        super(FeedbackLayer, self).__init__()

        self.name = name
        self.batch_dimensions = batch_dimensions
        self.allow_bp = allow_bp
        self._record_feedback_point = record_feedback_point

        self.feedback_point = None

    @property
    def record_feedback_point(self):
        return self._record_feedback_point

    @record_feedback_point.setter
    def record_feedback_point(self, value):
        self._record_feedback_point = value

        # If not storing the feedback points, free-up the memory of the feedback point:
        if not self._record_feedback_point:
            self.feedback_point = None

    def forward(self, input):
        # Feedback points are useful for backward calculations, only store them if we are calculating gradients:
        if input.requires_grad and self.record_feedback_point:  # TODO: input may be a tuple!
            self.feedback_point = input

        # Allow BP is used when reproducing the network but training with BP for alignment measurements:
        if self.allow_bp:
            return input
        else:
            input = input.detach()  # Cut the computation graph so that gradients don't flow back beyond FeedbackLayer
            input.requires_grad = True  # Gradients will still be required above
            return input

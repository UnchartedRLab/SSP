import torch
import torch.nn as nn

from ..dfa import DFAManager


class GradientAlignmentMetrics:
    def __init__(
        self,
        model,
        module_whitelist=None,
        generate_grad_function=None,
        metrics_device=None,
        sensitivity=1e-8,
        deterministic=True,
        strict_matching=False,
    ):
        # TODO: detect improper distributed use automagically.
        # Browse through DFA model and find the backend:
        self.model = model
        self.dfa_manager = GradientAlignmentMetrics._find_dfa_manager(self.model)

        self.module_whitelist = module_whitelist  # Will only measure alignment on these modules.
        self.generate_grad_function = generate_grad_function  # Executed to generate gradients in network.
        self.metrics_device = metrics_device  # Where to do the cosine similarity computations.
        self.sensitivity = sensitivity
        self.deterministic = deterministic  # Make sure DFA/BP gradients are taken with same RNG state.
        self.strict_matching = strict_matching  # Require all modules found in BP to exist in DFA as well

        self.grad_buffer = {}  # Temporary storage for gradients as they generated.
        self.grad_dfa = {}
        self.grad_bp = {}
        self.hooks_registry = []  # Temporary storage for hooks on the network, to disable them later.

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.sensitivity)

        # This dictionnary connects modules to their names in the model: easier to work with names for users.
        self.model_modules = {module: name for name, module in self.model.named_modules()}

    def __call__(self, data=None, target=None, loss_function=None):
        return self.measure_alignment(data, target, loss_function)

    def measure_alignment(self, data=None, target=None, loss_function=None):
        # First, check the arguments are valid: either there is a generate_grad_function,
        # or create a own basic one from the arguments provided.
        args_none_signature = [data is None, target is None, loss_function is None]
        if (self.generate_grad_function is None) and any(args_none_signature):
            raise ValueError(
                "No external generate_grad_function provided for alignment measurement, "
                "and no data/target/loss_function provided for manual definition! "
                "Provide either to enable alignment measurement."
            )
        elif not any(args_none_signature):
            # generate_grad_function hasn't been provided, so create one as a closure from the parameters provided:
            def internal_generate_grad_function(model):
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()

            self.generate_grad_function = internal_generate_grad_function  # Only temporary, will delete later.

        original_manager_configuration = (
            self.dfa_manager.no_feedbacks,
            self.dfa_manager.allow_bp_all,
            self.dfa_manager.record_feedback_point_all,
        )

        self.enable_alignment_measurement()  # Put the hooks into the network.
        # Collect the DFA gradients:
        with torch.random.fork_rng(enabled=self.deterministic):
            # The use of fork_rng ensures DFA and BP gradients will be generated with same RNG state.
            self.record_grad(self.grad_dfa)
            self.dfa_manager.use_bp = not self.dfa_manager.use_bp  # Ready the network for BP.
            # (We invert the state to allow measurements in a BP network as well.)

        self.record_grad(self.grad_bp)

        # Get the network back to where it was before measurement:
        self.dfa_manager.use_bp = not self.dfa_manager.use_bp
        (
            self.dfa_manager.no_feedbacks,
            self.dfa_manager.allow_bp_all,
            self.dfa_manager.record_feedback_point_all,
        ) = original_manager_configuration

        self.disable_alignment_measurement()  # Done with gradients collection, remove the hooks.

        alignment = self.evaluate_alignment()  # Actually evaluate alignment from the collected gradients.

        if not any(args_none_signature):
            self.generate_grad_function = None  # If a temporary function was used to get gradients, discard it.

        return alignment

    def enable_alignment_measurement(self):
        # TODO: add measure alignment method to DFA manager/RPoperation? (to ready things)
        # If using a whitelist, check each module against it before registering a hook on it:

        if self.module_whitelist is not None:
            for i, module in enumerate(self.model.modules()):
                if module in self.module_whitelist:
                    # TODO: as per torch doc, register_backward_hook will soon be deprecated in favor of register_full_backward_hook
                    self.hooks_registry.append(module.register_backward_hook(self.hook))  # Hook and record the hook.
        # Otherwise, hook into every module that is not made of submodules (for readability):
        else:
            for i, module in enumerate(self.model.modules()):
                if len(list(module.modules())) == 1:
                    self.hooks_registry.append(module.register_backward_hook(self.hook))  # Hook and record the hook.

    def disable_alignment_measurement(self):
        # Remove all hooks recorded.
        #self.dfa_manager.rp.sigma_privacy = self.ref_sigma_privacy
        #self.dfa_manager.rp.tau_feedback_privacy = self.tau_feedback_privacy
        for hook in self.hooks_registry:
            hook.remove()
        self.hooks_registry = []  # Reset the registry.

    def hook(self, module, grad_input, grad_output):
        # The actual backward hook that receives module and gradient info from PyTorch's autograd.
        self.grad_buffer[module] = grad_output[0]  # Save the info in the temporary buffer, don't know if DFA/BP yet.

    def record_grad(self, record_holder):
        self.generate_grad_function(self.model)  # Get the network to generate gradients.
        # Go through the buffer, move the gradients to a different device if needed, and add them to the DFA/BP dict.
        for module, grad in self.grad_buffer.items():
            if self.metrics_device is None:  # TODO: probably breaks distributed.
                self.metrics_device = grad.device
            # In buffer, we stored with module instead of name to avoid fetching the name in the hook.
            record_holder[self.model_modules[module]] = grad.detach().to(self.metrics_device)
        self.grad_buffer = {}  # Reset the buffer.
        self.model.zero_grad()  # Reset the model gradients.

    def evaluate_alignment(self):
        angles, alignments = {}, {}

        # BP/DFA gradients have been collected, now measure their alignments, module by module.
        for module in self.grad_dfa.keys():
            if module in self.grad_dfa and module in self.grad_bp:
                grad_dfa = self.grad_dfa[module]
                grad_bp = self.grad_bp[module]

                # Set small values to zero to avoid numerical instabilities.
                grad_dfa[grad_dfa.abs() <= self.sensitivity] = 0
                grad_bp[grad_bp.abs() <= self.sensitivity] = 0

                # Condition the gradients to avoid small values
                grad_dfa = grad_dfa.contiguous().view(grad_dfa.shape[0], -1) / self.sensitivity
                grad_bp = grad_bp.contiguous().view(grad_bp.shape[0], -1) / self.sensitivity

                angle = self.cosine_similarity(grad_bp, grad_dfa)  # Actually measure the cosine similarity/angles.
                angles[module] = angle
                alignment = [float(angle.mean()), float(angle.std())]  # Compile some stats on them for easy use.
                alignments[module] = alignment
            elif self.strict_matching:
                raise RuntimeError(
                    f"Module {module} could not be found in either DFA or BP model during gradient "
                    f"collection for alignment measurement!"
                )

        self.grad_dfa = {}
        self.grad_bp = {}

        return angles, alignments

    @staticmethod
    def _find_dfa_manager(model):
        for module in model.modules():
            if isinstance(module, DFAManager):
                return module
        raise ValueError(f"No DFAManager layer found in model {model}, cannot measure gradient alignment!")

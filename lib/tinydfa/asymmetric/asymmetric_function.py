import torch


class AsymmetricFunction:
    def __init__(self, forward_function, backward_function):
        self.forward_function = forward_function
        self.backward_function = backward_function

        class AutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return self.forward_function(input)

            @staticmethod
            def backward(ctx, grad_output):
                (input,) = ctx.saved_tensors
                return self.backward_function(input, grad_output)

        self.autograd_function = AutogradFunction.apply

    def __call__(self, input):
        return self.autograd_function(input)

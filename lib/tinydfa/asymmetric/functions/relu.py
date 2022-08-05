def relu_forward(input):
    return input.clamp(min=0)


def relu_backward(input, grad_output):
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input

import numpy as np
from activation_layer import ActivationLayer

class Sigmoid(ActivationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        def sigmoid_derivative(x):
            h = sigmoid(x)
            return h * (1 - h)
        super().__init__(sigmoid, sigmoid_derivative)
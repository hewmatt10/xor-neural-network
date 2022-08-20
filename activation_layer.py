from layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_function_derivative):
        self.f = activation_function
        self.f_prime = activation_function_derivative
    
    def forward_propagation(self, input):
        self.input = input
        return self.f(self.input)
    
    def backward_propagation(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.f_prime(self.input))
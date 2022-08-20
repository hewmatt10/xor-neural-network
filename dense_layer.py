from layer import Layer
import numpy as np

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)

    def forward_propagation(self, input):
        self.input = input
        return np.matmul(self.weights, self.input) + self.bias 
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.matmul(output_gradient, np.transpose(self.weights))
        self.weights -= learning_rate * np.matmul(output_gradient, np.transpose(self.input))
        self.bias -= learning_rate * output_gradient
        return input_gradient

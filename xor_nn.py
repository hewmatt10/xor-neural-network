import numpy as np
from dense_layer import DenseLayer
from activation_layer import ActivationLayer
from activation_functions import Sigmoid 
from loss_functions import mse, mse_prime 

X = np.array([[[0],[0]], [[0], [1]], [[1], [0]], [[1], [1]]])
Y = np.array([[[0]], [[1]], [[1]], [[0]]])

model = [
    DenseLayer(2, 3),
    Sigmoid(),
    DenseLayer(3, 1),
    Sigmoid(),
]

epochs = 100000
alpha = 0.1

for epoch in range(epochs):
    delta = 0
    for x, y in zip(X, Y):
        output = x 
        for layer in model: 
            output = layer.forward_propagation(output)
        delta += mse(y, output) 
        gradient = mse_prime(y, output)
        for layer in reversed(model):
            gradient = layer.backward_propagation(gradient, alpha)
    delta /= len(X)
    print(f'Error in epoch {epoch + 1}: {delta}')
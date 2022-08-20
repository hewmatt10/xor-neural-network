import numpy as np
from dense_layer import DenseLayer
from activation_layer import ActivationLayer
from activation_functions import Sigmoid 
from loss_functions import mse, mse_prime 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_train = np.array([[[0],[0]], [[0], [1]], [[1], [0]], [[1], [1]]])
Y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

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
    for x, y in zip(X_train, Y_train):
        output = x 
        for layer in model: 
            output = layer.forward_propagation(output)
        delta += mse(y, output) 
        gradient = mse_prime(y, output)
        for layer in reversed(model):
            gradient = layer.backward_propagation(gradient, alpha)
    delta /= len(X_train)
    print(f'Error in epoch {epoch + 1}: {delta}')

SPLITS = 20
X_test = np.array([[[i / SPLITS], [j / SPLITS]] for i in range(SPLITS+1) for j in range(SPLITS+1)])
Y_test = np.empty(shape=((SPLITS+1) * (SPLITS+1), 1, 1))
for i, x in enumerate(X_test):
    output = x 
    for layer in model: 
        output = layer.forward_propagation(output)
    Y_test[i] = output

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for x, y in zip(X_test, Y_test):
    ax.scatter(x[0], x[1], y[0])

plt.show()
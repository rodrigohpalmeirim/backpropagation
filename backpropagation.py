import numpy as np
import matplotlib.pyplot as plt
from data_generator import *

activation_functions = {
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0, x),
    "linear": lambda x: x
}
    
activation_derivatives = {
    "sigmoid": lambda x: activation_functions["sigmoid"](x) * (1 - activation_functions["sigmoid"](x)),
    "tanh": lambda x: 1 - np.power(activation_functions["tanh"](x), 2),
    "relu": lambda x: np.where(x > 0, 1, 0),
    "linear": lambda x: 1
}

loss_functions = {
    "mse": lambda y_true, y_pred: np.mean(np.power(y_true - y_pred, 2)),
    "cross_entropy": lambda y_true, y_pred: np.sum(y_true * np.log(y_pred))
}

loss_derivatives = {
    "mse": lambda y_true, y_pred: 2 * (y_true - y_pred),
    "cross_entropy": lambda y_true, y_pred: y_pred - y_true # TODO: fix this
}

class Layer:
    def __init__(self, n_inputs, n_neurons, learning_rate, activation):
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_inputs, n_neurons)-0.5
        self.biases = np.random.rand(n_neurons)-0.5
        self.inputs = None
        self.sum = None
        self.outputs = None
        self.weights_gradient = np.zeros((n_inputs, n_neurons))
        self.biases_gradient = np.zeros(n_neurons)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.sum = np.dot(inputs, self.weights) + self.biases
        self.outputs = activation_functions[self.activation](self.sum)
        return self.outputs

    def backward_pass(self, delta):
        d_sum = activation_derivatives[self.activation](self.sum) * delta
        self.weights_gradient = np.dot(self.inputs.T, d_sum)
        self.biases_gradient = np.sum(d_sum, axis=0)
        delta = np.dot(d_sum, self.weights.T)
        return delta

    def apply_gradients(self):
        self.weights += self.learning_rate * self.weights_gradient
        self.biases += self.learning_rate * self.biases_gradient
        self.weights_gradient = np.zeros((self.n_inputs, self.n_neurons))
        self.biases_gradient = np.zeros(self.n_neurons)

class Network:
    def __init__(self, loss, layers):
        self.layers = layers
        self.loss = loss
        self.training_losses = []

    def forward_pass(self, inputs, expected):
        for layer in self.layers:
            layer.forward_pass(inputs)
            inputs = layer.outputs
        outputs = inputs
        loss = loss_functions[self.loss](expected, outputs)
        return outputs, loss

    def backward_pass(self, expected, outputs):
        delta = loss_derivatives[self.loss](expected, outputs)
        for layer in reversed(self.layers):
            delta = layer.backward_pass(delta)

    def apply_gradients(self):
        for layer in self.layers:
            layer.apply_gradients()
    
    def train(self, minibatch_size, dataset, labels, epochs):
        for epoch in range(epochs):
            for i in range(0, len(dataset), minibatch_size):
                outputs, loss = self.forward_pass(dataset[i:i+minibatch_size], labels[i:i+minibatch_size])
                print("Epoch: {}, Loss: {}".format(epoch, loss), end="\r")
                self.training_losses.append(loss)
                self.backward_pass(labels[i:i+minibatch_size], outputs)
                self.apply_gradients()
    
    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

n = Network("mse", [
    Layer(n_inputs=16**2, n_neurons=10, learning_rate=0.1, activation="sigmoid"),
    Layer(n_inputs=10, n_neurons=10, learning_rate=0.1, activation="sigmoid"),
    Layer(n_inputs=10, n_neurons=4, learning_rate=0.1, activation="sigmoid")
])

d = Dataset(50000, 16, [Ellipse, Rectangle, Triangle, Cross], 0.5, 0.5, 0.05, 0.02, True)

n.train(minibatch_size=100, dataset=d.training_set, labels=d.labels, epochs=5)

plt.plot(n.training_losses)
plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.show()
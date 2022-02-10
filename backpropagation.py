import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
from functions import *

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
        self.weights_gradient = np.zeros((n_inputs, n_neurons))
        self.biases_gradient = np.zeros(n_neurons)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.sum = np.dot(inputs, self.weights) + self.biases
        return activation_functions[self.activation](self.sum)

    def backward_pass(self, delta):
        d_sum = activation_derivatives[self.activation](self.sum) * delta
        self.weights_gradient = np.dot(self.inputs.T, d_sum)
        self.biases_gradient = np.sum(d_sum, axis=0)
        return np.dot(d_sum, self.weights.T)

    def apply_gradients(self):
        self.weights += self.learning_rate * self.weights_gradient
        self.biases += self.learning_rate * self.biases_gradient
        self.weights_gradient = np.zeros((self.n_inputs, self.n_neurons))
        self.biases_gradient = np.zeros(self.n_neurons)

class SoftmaxLayer(Layer):
    def __init__(self, n_neurons):
        super().__init__(n_neurons, n_neurons, 0, "softmax")
        self.weights = None
        self.biases = None

    def forward_pass(self, inputs):
        self.inputs = inputs
        return activation_functions[self.activation](inputs)

    def backward_pass(self, delta):
        jacobian = activation_derivatives[self.activation](self.inputs)
        return np.einsum('ij,ijk->ik', delta, jacobian)

    def apply_gradients(self):
        pass

class Network:
    def __init__(self, loss, layers):
        self.layers = layers
        self.loss = loss
        self.training_losses = []
        self.validation_losses = []

    def forward_pass(self, inputs, expected):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
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
    
    def train(self, minibatch_size, dataset, epochs):
        for epoch in range(epochs):
            for i in range(0, len(dataset.training_set), minibatch_size):
                outputs, training_loss = self.forward_pass(dataset.training_set[i:i+minibatch_size], dataset.training_labels[i:i+minibatch_size])
                self.training_losses.append(training_loss)
                self.backward_pass(dataset.training_labels[i:i+minibatch_size], outputs)
                self.apply_gradients()

                _, validation_loss = self.forward_pass(dataset.validation_set, dataset.validation_labels)
                self.validation_losses.append(validation_loss)

                print(f"Epoch: {epoch+1}, TL: {training_loss:.3f}, VL: {validation_loss:.3f}", end="\r")

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

n = Network("cross_entropy", [
    Layer(n_inputs=16**2, n_neurons=10, learning_rate=0.01, activation="relu"),
    Layer(n_inputs=10, n_neurons=10, learning_rate=0.01, activation="relu"),
    Layer(n_inputs=10, n_neurons=4, learning_rate=0.01, activation="relu"),
    SoftmaxLayer(n_neurons=4)
])

d = Dataset(5000, 16, [Ellipse, Rectangle, Triangle, Cross], 0.5, 0.5, 0.05, 0.02, 0.7, 0.1, 0.1, True)

n.train(minibatch_size=100, dataset=d, epochs=20)
print()

_, test_loss = n.forward_pass(d.test_set, d.test_labels)
print("Test set loss:", test_loss)

plt.plot(n.training_losses, label="Training Loss")
plt.plot(n.validation_losses, label="Validation Loss")
plt.hlines(test_loss, xmin=len(n.training_losses), xmax=len(n.training_losses)+50, colors="green", label="Test Loss")
plt.legend()
plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.show()
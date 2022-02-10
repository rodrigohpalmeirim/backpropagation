import numpy as np
import matplotlib.pyplot as plt
import configparser
from data_generator import *
from functions import *
from sys import argv

class Layer:
    def __init__(self, n_inputs, n_neurons, weight_range, bias_range, learning_rate, activation):
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-weight_range, weight_range, (n_inputs, n_neurons))
        self.biases = np.random.uniform(-bias_range, bias_range, n_neurons)
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
        super().__init__(n_neurons, n_neurons, 0, 0, 0, "softmax")
        self.weights = None
        self.biases = None

    def forward_pass(self, inputs):
        self.inputs = inputs
        return softmax(inputs)

    def backward_pass(self, delta):
        jacobian = softmax_derivative(self.inputs)
        return np.einsum("ij,ijk->ik", delta, jacobian)

    def apply_gradients(self):
        pass

class Network:
    def __init__(self, loss, layers=[]):
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

                print(f"Epoch: {epoch+1}, Training loss: {training_loss:.3f}, Validation loss: {validation_loss:.3f}", end="\r")
        print()

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python3 backpropagation.py <config_file>")
        exit(1)

    config = configparser.ConfigParser()
    config.read(argv[1])

    # Create network
    n = Network(config["GLOBALS"]["loss_function"])

    # Add layers
    last_layer_neurons = int(config["DATASET"]["image_size"])**2
    for l in config["LAYERS"].values():
        params = eval(l)
        if "type" in params and params["type"] == "softmax":
            n.layers.append(SoftmaxLayer(last_layer_neurons))
        else:
            n.layers.append(Layer(
                n_inputs=last_layer_neurons,
                n_neurons=params["neurons"] if "neurons" in params else int(config["GLOBALS"]["neurons"]),
                weight_range=params["weight_range"] if "weight_range" in params else float(config["GLOBALS"]["weight_range"]),
                bias_range=params["bias_range"] if "bias_range" in params else float(config["GLOBALS"]["bias_range"]),
                learning_rate=params["learning_rate"] if "learning_rate" in params else float(config["GLOBALS"]["learning_rate"]),
                activation=params["activation"] if "activation" in params else config["GLOBALS"]["activation"]
            ))
            last_layer_neurons = params["neurons"]

    # Generate dataset
    d = Dataset(
        int(config["DATASET"]["dataset_size"]),
        int(config["DATASET"]["image_size"]),
        eval(config["DATASET"]["shapes"]),
        float(config["DATASET"]["size_variation"]),
        float(config["DATASET"]["pos_variation"]),
        float(config["DATASET"]["outline"]),
        float(config["DATASET"]["noise_amount"]),
        float(config["DATASET"]["training_ratio"]),
        float(config["DATASET"]["validation_ratio"]),
        float(config["DATASET"]["test_ratio"]),
        flatten=True
    )

    if bool(config["DATASET"]["show_dataset"]):
        d.show_data()

    # Train network
    n.train(minibatch_size=int(config["GLOBALS"]["minibatch_size"]), dataset=d, epochs=int(config["GLOBALS"]["epochs"]))

    # Test network
    _, test_loss = n.forward_pass(d.test_set, d.test_labels)
    print("Test set loss:", test_loss)

    # Plot losses
    plt.figure()
    plt.plot(n.training_losses, label="Training Loss")
    plt.plot(n.validation_losses, label="Validation Loss")
    plt.hlines(test_loss, xmin=len(n.training_losses), xmax=len(n.training_losses)+50, colors="green", label="Test Loss")
    plt.legend()
    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    plt.show(block=True)
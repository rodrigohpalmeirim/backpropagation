import numpy as np

def softmax(x):
    max_x = np.amax(x, 1).reshape(x.shape[0],1)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
    temp1 = np.einsum('ij,jk->ijk',s,a)
    temp2 = np.einsum('ij,ik->ijk',s,s)
    return temp1 - temp2
    
activation_functions = {
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0, x),
    "linear": lambda x: x,
    "softmax": softmax
}
    
activation_derivatives = {
    "sigmoid": lambda x: activation_functions["sigmoid"](x) * (1 - activation_functions["sigmoid"](x)),
    "tanh": lambda x: 1 - np.power(activation_functions["tanh"](x), 2),
    "relu": lambda x: np.where(x > 0, 1, 0),
    "linear": lambda x: 1,
    "softmax": softmax_derivative
}

loss_functions = {
    "mse": lambda y_true, y_pred: np.mean(np.power(y_true - y_pred, 2)),
    "cross_entropy": lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred))
}

loss_derivatives = {
    "mse": lambda y_true, y_pred: 2 * (y_true - y_pred),
    "cross_entropy": lambda y_true, y_pred: y_true / y_pred
}
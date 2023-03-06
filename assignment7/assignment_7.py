import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


def perceptron(x, iw, bias):
    return np.dot(x, iw) + bias

def sigmoid(x):
    #  The sigmoid activation function
    return 1 / (1 + np.exp(-x))


def loss(y_true, y_pred):
    # Mean squared error loss function
    return np.mean((y_true - y_pred)**2)

def forward_pass(x, iw, ow, bias):
    hidden_layer = sigmoid(perceptron(x, iw, bias))
    output_layer = np.dot(hidden_layer, ow)
    return hidden_layer, output_layer

def train(x_train, y_train, lr, epochs):
    input_weight = np.random.rand(2, 2)
    output_weight = np.random.rand(2)
    input_bias = np.random.rand(2)
    output_bias = np.random.rand(1)
    for i in range(epochs):
        hidden_layer, output_layer = forward_pass(x_train, input_weight, output_weight, input_bias)

        # chat
        d_output = 2 * (output_layer - y_train)
        d_weights_output = np.dot(hidden_layer.T, d_output)
        d_bias_output = np.sum(d_output)

        d_hidden = np.dot(d_output, output_weight.T) * hidden_layer * (1 - hidden_layer)
        d_weights_hidden = np.dot(x_train.T, d_hidden)
        d_bias_hidden = np.sum(d_hidden, axis=0)

        input_weight -= lr * d_weights_hidden
        input_bias -= lr * d_bias_hidden
        output_weight -= lr * d_weights_output
        output_bias -= lr * d_bias_output

        _, y_pred = forward_pass(x_train)
        print("Epoch:", i, "Loss:", loss(y_train, y_pred))

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    train(X_train, y_train, 0.01, 1)
    # TODO: Your code goes here.

import numpy as np
import matplotlib.pyplot as plt

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
    """

    :param x: input values going through the forward pass
    :param iw: input weight for used for hidden layer
    :param ow: output weight, for the output layer
    :param bias:
    :return: returns the hidden layer, and output layer
    """
    hidden_layer = sigmoid(perceptron(x, iw, bias))
    output_layer = np.dot(hidden_layer, ow)
    return hidden_layer, output_layer


def train(x_train, y_train, lr, epochs, x_test=None):
    # input data has two dimensions, we have two hidden components
    input_dim = 2
    hidden_dim = 2
    # we will get one value as output
    output_dim = 1

    # weights and biases are defined by their corresponding dimensions
    # will need to be of compatible dimensions for further calculation with input/output
    input_weight = np.random.normal(size=(input_dim, hidden_dim))
    output_weight = np.random.normal(size=(input_dim, output_dim))
    # biases are a simple random variable
    input_bias = np.random.normal(size=(1, hidden_dim))
    output_bias = np.random.normal(size=(1, output_dim))
    for i in range(epochs):
        # running forward pass for the training data with corresponding weigts and biases
        hidden_layer, output_layer = forward_pass(x_train, input_weight, output_weight, input_bias)

        # some issues with shapes when taking the dot product,
        # therefore reshaping one dimensional array to (280, 1)
        y_train = y_train.reshape((280, 1))

        # gradient calculatation where derivative output is 2 * (output layer - training output)
        d_output = 2 * (output_layer - y_train)
        # weights are calculated using dot pruduct of hidden layer and derived output
        d_weights_output = np.dot(hidden_layer.T, d_output)
        d_bias_output = np.sum(d_output)

        # calculate new hidden component with weights and biases using dotproduct and factoring the hidden layer
        # derivative of the loss function with inputs of a hidden layer previously calculated using the sigmoid function
        # this is the backpropagation component of the algorithm
        d_hidden = np.dot(d_output, output_weight.T) * hidden_layer * (1 - hidden_layer)
        d_weights_hidden = np.dot(x_train.T, d_hidden)
        d_bias_hidden = np.sum(d_hidden, axis=0)

        # adjust the weights, decrease corresponding to learning rate
        input_weight -= lr * d_weights_hidden
        input_bias -= lr * d_bias_hidden
        output_weight -= lr * d_weights_output
        output_bias -= lr * d_bias_output

        _, y_pred = forward_pass(x_train, input_weight, output_weight, input_bias)
        if (i+1) % 10000 == 0:
            print("Epoch:", i+1, "Loss:", loss(y_train, y_pred))

    if x_test is not None:
        # if we send in test data, we will return the corresponding predictions
        print("PREDICTING TEST DATA")
        _, y_pred = forward_pass(x_test, input_weight, output_weight, input_bias)

    return y_pred

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    y_pred = train(X_train, y_train, 0.0001, 100000, x_test=X_test)

    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.show()
    plt.scatter(range(len(y_test)), y_test)
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.xlabel('Index')
    plt.ylabel('Predictions')
    plt.show()

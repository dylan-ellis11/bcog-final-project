# mlp_classifier.py outline

import numpy as np

def load_and_preprocess_data(filepath):
    """
    (Corresponds to user's 'function_loading')
    Loads raw image data from an .npz file or directory.
    - Converts to grayscale (if RGB).
    - Resizes to uniform pixel dimensions (28x28).
    - Flattens 2D arrays into 1D arrays (size 784).
    - Normalizes pixel values by dividing by 255.0.
    Returns: X_train, y_train, X_test, y_test
    """
    pass

class MultiLayerPerceptron:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes):
        """
        Initializes the 3-layer network architecture.
        Sets up the weight matrices and bias vectors with random small values.
        """
        pass
        
    def _sigmoid(self, x):
        """Activation function for the layers."""
        pass

    def _sigmoid_derivative(self, x):
        """Calculates the derivative for backpropagation calculus."""
        pass

    def feed_forward(self, X):
        """
        (Corresponds to user's 'function_feed_forward')
        Propagates the input data through the 3 layers.
        Calculates weighted sums and applies the activation function.
        Returns the final probability score / classification guess.
        """
        pass

    def backpropagate(self, X, y, output_guess, learning_rate):
        """
        (Corresponds to user's 'function_train' core logic)
        Calculates the gradient of the loss function using the chain rule.
        Adjusts the weights and biases (synaptic strengths) via gradient descent.
        """
        pass

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Loops through the dataset for a given number of epochs.
        Calls feed_forward and backpropagate to train the network iteratively.
        Prints loss metrics to track diagnostic error reduction.
        """
        pass

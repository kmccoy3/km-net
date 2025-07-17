#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python module containing a basic implementation of a neural network.
"""

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================

__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2025, McCoy"
__credits__ = ["Kevin McCoy"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Kevin McCoy"
__email__ = ["kevin@kmccoy.net"]
__status__ = "development"
__date__ = "2025-07-16" # Last modified date

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def relu(x):
    """Applies the ReLU activation function."""

    return np.maximum(0, x)

# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================

class nn:
    """A simple neural network / multilayer perceptron class."""

    def __init__(self, layer_widths):
        """ Initializes the neural network with given layer widths.  
        Args:
            layer_widths (list): A list of integers representing the number of neurons in each 
                                 layer. This includes the input layer, hidden layers, and output 
                                 layer.
        """

        self.layer_widths = layer_widths
        
        self.num_layers = len(layer_widths) - 1
        self.depth = self.num_layers
        self.input_size = layer_widths[0]
        self.output_size = layer_widths[-1]

        # Initialize weights and biases
        self.randomize_layers()


    def randomize_layers(self):
        """Randomizes the weights and biases of the neural network."""

        self.W = {}
        self.b = {}
        for k in range(1, self.depth + 1):
            self.W[k] = np.random.uniform(-1, 1, size=(self.layer_widths[k], self.layer_widths[k-1]))
            self.b[k] = np.random.uniform(-1, 1, size=(self.layer_widths[k], 1))
    

    def forward(self, x):
        """Performs a forward pass through the network.
        Args:
            x (numpy.ndarray): Input data of shape (input_size, num_samples).
        Returns:
            numpy.ndarray: Output of the network after the forward pass.
        """

        self.h = {0: x}  # Store activations for backpropagation
        self.a = {}

        # Loop through each layer to compute activations
        for k in range(1, self.num_layers + 1):
            a = np.matmul(self.W[k], self.h[k-1]) + self.b[k]
            self.a[k] = a
            if k < self.num_layers:
                self.h[k] = relu(a)
            else:
                self.h[k] = a # No activation for output layer

        return self.h[self.num_layers]
    

    def zero_gradients(self):
        """Initializes gradients to zero for backpropagation."""

        self.dW = {}
        self.db = {}
        for k in range(1, self.num_layers + 1):
            self.dW[k] = np.zeros_like(self.W[k])
            self.db[k] = np.zeros_like(self.b[k])


    def backprop(self, y):
        """Performs backpropagation to compute gradients.
        Args:
            y (numpy.ndarray): True labels of shape (output_size, num_samples).
        """

        # Compute the gradient of the loss with respect to the output
        g = self.h[self.num_layers] - y

        # Backpropagate through the network
        for k in range(self.num_layers, 0, -1):

            # Compute the gradient of the loss with respect to the activation
            if k != self.num_layers:
                g = np.multiply(g, self.a[k]>0)

            # Compute the gradients for weights and biases
            self.dW[k] += np.matmul(g, self.h[k-1].T)
            self.db[k] += g
            
            # Compute the gradient for the previous layer
            g = np.matmul(self.W[k].T, g)


    def update(self, learning_rate):
        """Updates the weights and biases using the computed gradients.
        Args:
            learning_rate (float): Learning rate for the update.
        """

        # Loop through each layer and update weights and biases
        for k in range(1, self.num_layers + 1):
            self.W[k] -= learning_rate * self.dW[k]
            self.b[k] -= learning_rate * self.db[k]


    def train(self, x, y, num_epochs=100, learning_rate=[0.1, 0.01, 0.001], plot_progress=False):
        """Trains the neural network using the provided data.
        Args:
            x (numpy.ndarray): Input data of shape (input_size, num_samples).
            y (numpy.ndarray): True labels of shape (output_size, num_samples).     
            num_epochs (int): Number of epochs to train the network.
            learning_rate (list): List of learning rates for different stages of training.
            plot_progress (bool): Whether to plot the training progress.
        """

        print("Training started...")
        
        # Loop through entire dataset
        for epoch in range(num_epochs):

            # Loop over each observation in the dataset
            for i in range(x.shape[1]):
                self.zero_gradients() # Initialize gradients to zero
                self.forward(x[:, i:i+1]) # Forward pass
                self.backprop(y[:, i:i+1]) # Backward pass
                stage = epoch // (num_epochs // len(learning_rate) + 1) # Use simple LR scheduler
                self.update(learning_rate[stage]) # Update weights and biases
            
            # Print loss and plot progress every 10 epochs
            if epoch % 10 == 0:
                preds = self.forward(x)
                loss = np.mean((preds - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.3f}')

                if plot_progress:
                    self.plot(x, y, preds, epoch)


    def plot(self, x, y, preds, epoch):
        """Plots the true and predicted values for the current epoch.
        Args:
            x (numpy.ndarray): Input data of shape (input_size, num_samples).
            y (numpy.ndarray): True labels of shape (output_size, num_samples).
            preds (numpy.ndarray): Predicted values of shape (output_size, num_samples).
            epoch (int): Current epoch number.
        """

        # Sort the data for better visualization
        order = np.argsort(x[0])
        x = x[:, order]
        y = y[:, order]
        preds = preds[:, order]

        # Plot the true and predicted values
        plt.plot(x.T, y.T, 'o')
        plt.plot(x.T, preds.T, '-')

        # Set labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Epoch #{epoch}')
        plt.legend(['True', 'Predicted'])

        # Save the plot
        plt.savefig(f'./images/epoch_{epoch}.png', transparent=False, facecolor='white')
        plt.close()


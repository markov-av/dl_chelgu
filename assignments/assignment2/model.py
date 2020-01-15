import numpy as np

from layers import FullyConnectedLayer, ReLULayer, \
    softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = (FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        for layer in self.layers:
            layer.gradients_reset()
        # Hint: using self.params() might be useful!
        for layer in self.layers:
            # print('     ', X.shape)
            X = layer.forward(X)
        # TODO Compute loss and fill param gradients
        loss, dprediction = softmax_with_cross_entropy(X, y)
        # by running forward and backward passes through the model
        for layer in reversed(self.layers):
            dprediction = layer.backward(dprediction)
        # After that, implement l2 regularization on all params
        if self.reg:
            for num, layer in enumerate(self.layers):
                if isinstance(layer, FullyConnectedLayer):
                    loss_by_l2, d_reg = l2_regularization(layer.W.value, self.reg)
                    loss += loss_by_l2
                    layer.W.grad += d_reg
        # Hint: self.params() is useful again!
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        return pred.argmax(axis=1)

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        for num_layer, layer in enumerate(self.layers):
            for key, value in layer.params().items():
                result[(num_layer, key)] = value

        return result

import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        return np.exp(predictions) / np.sum(np.exp(predictions))
    else:
        predictions -= np.amax(predictions, axis=1).reshape(predictions.shape[0], 1)
        return np.exp(predictions) / np.sum(np.exp(predictions), axis=1).reshape((predictions.shape[0], 1))


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W
    return loss, grad


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
        loss = -np.sum(np.log(probs[target_index]))
    else:
        loss = -np.sum(
            np.log(probs[np.arange(target_index.shape[0]), target_index.reshape(1, -1)]))
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    if predictions.ndim == 1:
        d_zeros_pred = np.zeros(predictions.shape)
        d_zeros_pred[target_index] = 1
    else:
        d_zeros_pred = np.zeros(predictions.shape)
        d_zeros_pred[np.arange(len(target_index)), target_index.reshape(1, -1)] = 1
    loss = cross_entropy_loss(softmax(predictions.copy()), target_index)
    dprediction = softmax(predictions.copy()) - d_zeros_pred
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.cache = (X > 0)
        relu = lambda x: x * (x > 0).astype(float)
        out = relu(X)
        return out

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        return self.cache * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

    def gradients_reset(self):
        pass


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        # It should be pretty similar to linear classifier from
        # the previous assignment

        dx = np.dot(d_out, self.W.value.T)
        dw = np.dot(self.X.T, d_out)
        self.W.grad = dw
        db = np.dot(np.ones(shape=(1, self.X.shape[0])), d_out)
        self.B.grad = db
        return dx

    def params(self):
        return {'W': self.W, 'B': self.B}

    def gradients_reset(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                pass
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass

        raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}

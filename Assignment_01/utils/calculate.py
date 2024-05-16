from builtins import range
import numpy as np


def linear_forward(x, w, b):
    """Computes the forward pass for an linear (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    num_examples = x.shape[0]

    flattend_x = x.reshape((num_examples, -1))
    out = flattend_x @ w + b

    cache = (x, w, b)
    return out, cache


def linear_backward(dout, cache):
    """Computes the backward pass for an linear (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    num_examples = x.shape[0]

    flattend_x = x.reshape((num_examples, -1))
    dw = flattend_x.T @ dout
    db = np.ones(num_examples) @ dout
    dx = dout @ w.T
    dx = dx.reshape(x.shape)

    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.where(x > 0, x, 0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    mask = (x > 0).astype(int)  # (N,...)
    dx = dout * mask

    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    logits = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    loss = -np.log(probs[range(len(y)), y])
    loss = np.mean(loss)

    probs[range(len(y)), y] -= 1
    dx = probs / len(y)

    return loss, dx


def linear_relu_forward(x, w, b):
    """Convenience layer that performs an linear transform followed by a ReLU.

    Inputs:
    - x: Input to the linear layer
    - w, b: Weights for the linear layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def linear_relu_backward(dout, cache):
    """Backward pass for the linear-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db



def linear_relu_dropout_forward(x, w, b, dropout_param):
    """Convenience layer that performs an linear transform followed by a ReLU.

    Inputs:
    - x: Input to the linear layer
    - w, b: Weights for the linear layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache

def linear_relu_dropout_backward(dout, cache):
    """Backward pass for the linear-relu convenience layer.
    """
    fc_cache, relu_cache, dropout_cache = cache
    dout = dropout_backward(dout, dropout_cache)
    da = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db

def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, mu, var, std, gamma, x_hat, shape = cache                  # expand cache

    dbeta = dout.reshape(shape, order='F').sum(axis=0)            # derivative w.r.t. beta
    dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis=0) # derivative w.r.t. gamma

    dx_hat = dout * gamma                                       # derivative w.t.r. x_hat
    dstd = -np.sum(dx_hat * (x-mu), axis=0) / (std**2)          # derivative w.t.r. std
    dvar = 0.5 * dstd / std                                     # derivative w.t.r. var
    dx1 = dx_hat / std + 2 * (x-mu) * dvar / len(dout)          # partial derivative w.t.r. dx
    dmu = -np.sum(dx1, axis=0)                                  # derivative w.t.r. mu
    dx2 = dmu / len(dout)                                       # partial derivative w.t.r. dx
    dx = dx1 + dx2                                              # full derivative w.t.r. x

    return dx, dgamma, dbeta


def linear_norm_relu_forward(normalization, x, w, b, gamma, beta, bn_param):
    """Convenience layer that performs an linear transform followed by BatchNorm and a ReLU.

    Inputs:
    - x: Input to the linear layer
    - w, b: Weights for the linear layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    if normalization=="batchnorm":
        a, norm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache, norm_cache)
    return out, cache


def linear_norm_relu_backward(normalization, dout, cache):
    """Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, relu_cache, norm_cache = cache
    da = relu_backward(dout, relu_cache)
    if normalization == "batchnorm":
        da, dgamma, dbeta = batchnorm_backward(da, norm_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
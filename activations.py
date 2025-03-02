#Applying randomly initialized neural networks to input datasets.
import numpy as np

#Parameters for a random layer of a neural network
def uniform_sphere_sample(dim):
    """
    Samples from the uniform distribution on the n-dimensional unit sphere.
    Args:
        dim: Dimension of the unit sphere to sample from.
    """
    normal_vec = np.random.standard_normal(dim)
    return normal_vec/np.linalg.norm(normal_vec)

def rand_layer_params(input, output, max_bias):
    """
    Randomly generates weights and biases of a layer of a neural network.
    Args:
        input: Input dimension of the data.
        output: Desired output dimension of the data.
        max_bias: Maximal bias of the layer. Determines what uniform distribution
        to sample from.
    """
    W = np.random.normal(0,1,(output,input))
    b = 2*max_bias*np.random.rand(output) - max_bias*np.zeros(output) #b from uniform [-rad, rad] distribution
    return (W,b)

#Activation Functions: ReLU and linear
def relu_layer(W,b,x):
    return np.maximum(W @ x + b, 0)

def linear_layer(W,b,x):
    return W @ x + b

#Random Networks
def apply_rand_layer(X, output, max_bias, layer_type):
    """
    Applies a random layer of a neural network to an input dataset.
    Args:
        X: A matrix where each row is a datapoint.
        output: The desired output dimension.
        max_bias: Maximal bias of the random layer.
        layer_type: Activation function of the random layer.
    """
    newX = np.ndarray((len(X),output))
    W,b = rand_layer_params(len(X[0]), output, max_bias)
    for i in range(len(X)):
        newX[i] = layer_type(W,b,X[i])
    return newX * np.sqrt(2/output)

def one_layer_relu_net(X, width, max_bias, use_projection=False):
    """
    Applies a one layer neural network with ReLU activation to an input dataset.
    Args:
        X: A matrix where each row is a datapoint.
        output: The desired output dimension.
        max_bias: Maximal bias of the random layer.
        use_projection: If True, multiplies by a random matrix to reduce the dataset
        to 2D for visualization purposes. 
    """
    currentX = apply_rand_layer(X, width, max_bias, relu_layer)
    if use_projection:
        currentX = apply_rand_layer(currentX, 2, 0, linear_layer)
    return currentX
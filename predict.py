import numpy as np

#####################################
# from Udacity lesson Cross-Entropy 2
#####################################
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y  *np.log(P) + (1 - Y)*  np.log(1 - P))

#####################################
# from Udacity GradientDescentSolutions
#####################################

# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
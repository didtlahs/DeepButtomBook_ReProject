import numpy as np



def Sigmoid(x):
    val = (1 + np.exp(-x))
    return (1 / val)


def SoftMax(x):
    max = np.max(x)
    exp_val = np.exp(x - max)
    return exp_val / np.sum(exp_val)
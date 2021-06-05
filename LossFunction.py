import numpy as np

def MeanSquaredError(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def CrossEntropyError(y, t):
    delta = 1e-7
    return - np.sum(np.log(y + delta) * t)
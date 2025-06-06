import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

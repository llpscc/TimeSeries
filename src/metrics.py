import numpy as np


def nwrmsle(y_true, y_pred, weights):
    return np.sqrt(
        np.sum(weights * (np.log1p(y_pred) - np.log1p(y_true)) ** 2)
        / np.sum(weights)
    )
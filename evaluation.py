import numpy as np

_EPS = 1e-20


def _mae(real, score):
    return np.abs((real - score)).sum() / (len(real) + _EPS)


def _rmse(real, score):
    return np.sqrt(((real - score) ** 2).sum() / (len(real) + _EPS))


def evaluate(real, score):
    real = np.array(real)
    score = np.array(score)
    return {
        'mae': _mae(real, score),
        'rmse': _rmse(real, score)
    }

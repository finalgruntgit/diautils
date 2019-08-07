import numpy as np


def siglog(v):
    return np.sign(v) * np.log(1 + np.abs(v))


def sigexp(v):
    sv = np.sign(v)
    return sv * (np.exp(sv * v) - 1)


def z_norm(v, mean=None, std=None):
    if mean is None:
        mean = v.mean()
    if std is None:
        std = v.std()
    np.where(std == 0, 1, std)
    return (v - mean) / std


def siglog_norm(v, mean=None, std=None):
    return siglog(z_norm(v, mean, std))


def tanh_siglog_norm(v, mean=None, std=None):
    return np.tanh(siglog_norm(v, mean, std))

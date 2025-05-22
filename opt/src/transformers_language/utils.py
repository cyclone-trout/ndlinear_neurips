
import torch
import torch.nn as nn


def kurtosis(x, eps=1e-6):
    """x - (B, d)"""

    # print("Shape of the input tensor of kurtosis", x.shape)
    mu = x.mean(dim=1, keepdims=True)
    # print("mean of input tensor of kurtosis", mu)
    s = x.std(dim=1)
    # print("standard deviation of input tensor of kurtosis", s)
    mu4 = ((x - mu) ** 4.0).mean(dim=1)
    k = mu4 / (s**4.0 + eps)
    # print("kurtosis:", k)
    return k


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


class DotDict(dict):
    """
    This class enables access to its attributes as both ['attr'] and .attr .
    Its advantage is that content of its `instance` can be accessed with `.`
    and still passed to functions as `**instance` (as dictionaries) for
    implementing variable-length arguments.
    """

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")

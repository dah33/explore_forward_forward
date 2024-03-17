import torch.nn as nn
import torch.nn.functional as F


def goodness(h):
    """Goodness is the *mean* squared activation of a layer, which may be
    multi-dimensional."""
    return h.pow(2).flatten(1).mean(1)


class UnitLength(nn.Module):
    """Layer that normalises its (possibly multi-dimensional) inputs to a unit
    length vector. Do this to "conceal" goodness from the next layer."""

    def forward(self, x):
        original_shape = x.shape
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        x = x.view(original_shape)
        return x


class LayerOutputs:
    """
    Iterator that returns the output of each layer in a model, in turn. Model
    must be an iterable of layers.

    Example:
        >>> model = nn.Sequential(...)
        >>> [h.mean() for h in LayerOutputs(model, x)]
    """

    def __init__(self, model, x):
        self.layers = iter(model)
        self.x = x

    def __iter__(self):
        return self

    def __next__(self):
        layer = next(self.layers)
        self.x = layer(self.x)
        return self.x

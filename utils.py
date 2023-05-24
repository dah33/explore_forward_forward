import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class UnitLength(nn.Module):
    """Layer that normalises its inputs to a unit length vector"""
    def forward(self, x):
        return F.normalize(x)
    
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

def visualise_sample(x, title='', sample_index=0):
    img = x[sample_index].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.show()

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import split
from torch.optim import Adam
from utils import LayerOutputs, UnitLength

# %%
def distance2_to_centroids(h, y_true, epsilon=1e-12):
    """
    Calculates the mean squared distance to the centroid of each class. 

    Returns a tensor of shape [n_examples, 10].
    """
    safe_mean = lambda x, dim: x.sum(dim) / (x.shape[dim] + epsilon)
    # TODO: what if class is missing? determine centroids only for classes that are present, and return torch.unique(y_true)
    class_centroids = torch.stack([safe_mean(h[y_true == i],0) for i in range(10)], dim=1) # [n_in, 10]
    x_to_centroids = h.unsqueeze(2) - class_centroids # [n_examples, n_in, 10]
    return x_to_centroids.pow(2).mean(1) # [n_examples, 10]
    
@torch.no_grad()
def predict(model, x, y_true):
    """Predict by finding the class with closest centroid to each example."""
    d = sum(distance2_to_centroids(h, y_true) for h in LayerOutputs(model, x))
    return d.argmin(1) # type: ignore

# %%
def centroid_loss(h, y_true, alpha=10, epsilon=1e-12):
    """
    Loss function based on distance^2 to the true centroid vs a nearby centroid.
    
    Achieves an error rate of ~2.0%.
    """

    # Distance from h to centroids of each class
    d2 = distance2_to_centroids(h, y_true)

    # Choose a nearby class, at random, using the inverse distance as a
    # probability distribution
    y_near = torch.multinomial((1 / (d2 + epsilon)), 1).squeeze(1)

    # Smoothed version of triplet loss: max(0, d2_same - d2_near + margin)
    d2_true = d2[range(d2.shape[0]), y_true] # ||anchor - positive||^2
    d2_near = d2[range(d2.shape[0]), y_near] # ||anchor - negative||^2
    return F.silu(alpha * (d2_true - d2_near)).mean()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# The data is pre-processed, to speed up this script
x_tr = torch.load('./data/MNIST/baked/train_x.pt', device)
y_tr = torch.load('./data/MNIST/baked/train_y.pt', device)
x_te = torch.load('./data/MNIST/baked/test_x.pt', device)
y_te = torch.load('./data/MNIST/baked/test_y.pt', device)

# %% 
# Define the model
# ----------------
# Must be an iterable of layers. I find it works best if each layer starts with
# a UnitLength() sub-layer.
model = nn.Sequential(
    nn.Sequential(UnitLength(), nn.Linear(784, 500), nn.ReLU()),
    nn.Sequential(UnitLength(), nn.Linear(500, 500), nn.ReLU()),
).to(device)

# %%
# Evaluate the model on the training and test set
def print_evaluation(epoch=None):
    global model, x_tr, y_tr, x_te, y_te
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x, y), y)
    train_error = prediction_error(x_tr, y_tr)
    test_error = prediction_error(x_te, y_te)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"
    print(f"[{epoch_str}] Training: {train_error*100:>5.2f}%\tTest: {test_error*100:>5.2f}%")

# %%
# Training parameters
torch.manual_seed(42)
loss_fn = centroid_loss
learning_rate = 0.05
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 120+1
batch_size = 4096

# %%
# Train the model
print_evaluation()
for epoch in range(num_epochs):

    # Mini-batch training
    for x, y in zip(split(x_tr, batch_size), split(y_tr, batch_size)):

        # Train layers in turn, using backprop locally only
        for layer in model:
            h = layer(x)
            loss = centroid_loss(h, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            x = h.detach() # no need to forward propagate x again, as direction doesn't change

    # Evaluate the model on the training and test set
    if (epoch + 1) % 5 == 1:
        print_evaluation(epoch)

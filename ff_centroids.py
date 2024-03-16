# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import mnist
from ff_utils import LayerOutputs, UnitLength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def distance_to_centroids(h, y_true, epsilon=1e-12):
    """
    Calculates the mean squared distance to the centroid of each class.

    Returns a tensor of shape [n_examples, 10].
    """
    safe_mean = lambda x, dim: x.sum(dim) / (x.shape[dim] + epsilon)
    # TODO: what if class is missing?
    # * determine centroids only for classes that are present, and return torch.unique(y_true)
    # * or treat centroids as trainable parameters, so they slowly update
    class_centroids = torch.stack(
        [safe_mean(h[y_true == i], 0) for i in range(10)], dim=1
    )  # [n_in, 10]
    x_to_centroids = h.unsqueeze(2) - class_centroids  # [n_examples, n_in, 10]
    return x_to_centroids.pow(2).mean(1)  # [n_examples, 10]


@torch.no_grad()
def predict(model: nn.Sequential, x, y_true, skip_layers=1):
    """Predict by finding the class with closest centroid to each example."""
    d = sum(
        [distance_to_centroids(h, y_true) for h in LayerOutputs(model, x)][skip_layers:]
    )
    return d.argmin(1)  # type: ignore


def centroid_loss(h, y_true, alpha=4.0, epsilon=1e-12, temperature=1.0):
    """
    Loss function based on distance^2 to the true centroid vs a nearby centroid.

    Achieves an error rate of ~1.7%.
    """

    # Distance from h to centroids of each class
    d2 = distance_to_centroids(h, y_true)

    # Choose a nearby class, at random, using the inverse distance as a
    # probability distribution, normalised by the minimum distance to avoid
    # out-of-range values.
    min_d2 = torch.min(d2, 1, keepdim=True)[0]
    norm_d2 = (d2 + epsilon) / (min_d2 + epsilon)
    y_near = torch.multinomial(norm_d2.pow(-temperature), 1).squeeze(1)

    # Smoothed version of triplet loss: max(0, d2_same - d2_near + margin)
    d2_true = d2[range(d2.shape[0]), y_true]  # ||anchor - positive||^2
    d2_near = d2[range(d2.shape[0]), y_near]  # ||anchor - negative||^2
    return F.silu(alpha * (d2_true - d2_near)).mean()


# %%
# Define the model
#
# Must be an iterable of layers. I find it works best if each layer starts with
# a UnitLength() sub-layer.
n_units = 500  # 2000 improves error rate
model = nn.Sequential(
    nn.Sequential(UnitLength(), nn.Linear(784, n_units), nn.ReLU()),
    nn.Sequential(UnitLength(), nn.Linear(n_units, n_units), nn.ReLU()),
).to(device)


def error_rate(model: nn.Sequential, data_loader: DataLoader) -> float:
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        predicted = predict(model, x, y)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return 1 - correct / total


# %%
# Train the model
torch.manual_seed(42)
learning_rate = 0.05
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 120 + 1
batch_size = 4096
train_loader = DataLoader(
    list(zip(mnist.train_x, mnist.train_y)), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    list(zip(mnist.test_x, mnist.test_y)), batch_size=batch_size, shuffle=False
)

print(
    "[init] Training: {:.2%}, Test: {:.2%}".format(
        error_rate(model, train_loader),
        error_rate(model, test_loader),
    )
)
for epoch in range(num_epochs):

    # Mini-batch training
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Train layers in turn on same mini-batch, using backpropagation locally only
        model.train()
        for layer in model:
            h = layer(x)
            temperature = 4
            loss = centroid_loss(h, y, temperature=temperature)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                x = layer(x)

    # Evaluate the model on the training and test set
    if (epoch + 1) % 5 == 1:
        print(
            "[{:>4d}] Training: {:.2%}, Test: {:.2%}".format(
                epoch + 1,
                error_rate(model, train_loader),
                error_rate(model, test_loader),
            )
        )

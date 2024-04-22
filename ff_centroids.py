# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
from torch.optim import Adam
from torch.utils.data import DataLoader

import mnist
from ff_utils import LayerOutputs, UnitLength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def distance_to_centroids(
    h: torch.Tensor,
    y_true: torch.Tensor,
    return_ytrue: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean squared distance to the centroid of each class present in the batch
    and returns the remapped y_true with new class indices.

    Args:
        h: A tensor of shape (n_examples, n_features)
        y_true: A tensor of shape (n_examples,) giving true labels for the batch
        return_ytrue: Whether to return the remapped y_true with new class indices

    Returns:
        distances: A tensor of shape (n_examples, n_classes_in_batch)
        remapped_y_true: A tensor of shape (n_examples,) giving y_true remapped to new class indices

    Example usage with dummy data:
        >>> h = torch.rand(5, 2)  # 5 examples, 2 features each
        >>> y_true = torch.randint(0, 10, (5,))  # Random classes from 0 to 9 for each example
        >>> distances, remapped_y_true = distance_to_centroids(h, y_true, return_ytrue=True)
        >>> print(distances.shape, remapped_y_true, sep='\n')
        torch.Size([5, 3])
        tensor([2, 1, 1, 1, 0]) # values may vary
    """
    # Identify the unique classes present in the batch and their mapping
    classes_in_batch, remapped_y_true = torch.unique(
        y_true, sorted=True, return_inverse=True
    )

    # Calculate centroids for the classes present in the batch
    class_centroids = []
    for idx in range(classes_in_batch.size(0)):
        mask = remapped_y_true == idx
        class_mean = h[mask].mean(dim=0)
        class_centroids.append(class_mean)
    class_centroids = torch.stack(class_centroids, dim=1)

    # Compute distances from each example to its class centroid
    x_to_centroids = (
        h.unsqueeze(2) - class_centroids
    )  # [n_examples, n_features, n_classes_in_batch]
    distances = x_to_centroids.pow(2).mean(1)  # [n_examples, n_classes_in_batch]

    return (distances, remapped_y_true) if return_ytrue else distances


@torch.no_grad()
def predict(model: nn.Sequential, x, y_true, skip_layers=1):
    """
    Predict by finding the class with closest centroid to each example.

    TODO: If there's only one or a few examples of a class, the centroid will be
    very close to the example itself, a data leakage issue. This can be fixed by
    excluding the example from the centroid calculation, using the training
    centroids, or somehow using a different method to predict.
    """
    d = sum(
        [distance_to_centroids(h, y_true) for h in LayerOutputs(model, x)][skip_layers:]
    )
    return d.argmin(1)  # type: ignore


def centroid_loss(h, y_true, temperature=1.0):
    """
    Loss function based on (squared) distance to the true centroid vs other centroids.

    Achieves an error rate of ~2.2%.
    """

    # Distance from h to centroids of each class (in the batch)
    d, y_true = distance_to_centroids(h, y_true, return_ytrue=True)

    # Softmax then calculate the cross-entropy loss
    return F.cross_entropy(-d * math.exp(temperature), y_true, reduction="mean")


# %%
# Define the model
#
# Must be an iterable of layers. I find it works best if each layer starts with
# a UnitLength() sub-layer.
n_units = 500  # 2000 improves error rate
model = nn.Sequential(
    nn.Sequential(Flatten(), UnitLength(), nn.Linear(784, n_units), nn.ReLU()),
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
num_epochs = 120
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
    if epoch % 5 == 0:
        print(
            "[{:>4d}] Training: {:.2%}, Test: {:.2%}".format(
                epoch,
                error_rate(model, train_loader),
                error_rate(model, test_loader),
            )
        )

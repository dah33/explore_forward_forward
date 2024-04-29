# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
from torch.optim import Adam
from torch.utils.data import DataLoader

import mnist
from ff_utils import LayerOutputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def calculate_distance_matrix(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Return the cosine distance between each pair of vectors in x1 and x2.

    Args:
        x1 [N, D]
        x2 [M, D]

    Returns:
        [N, M] matrix of distance between each pair of vectors x1[i] and x2[j]
    """
    assert x1.size(1) == x2.size(1), "x1 and x2 must have the same number of features"
    return -F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=2)


def remap_class_labels(labels):
    """
    Remap the class labels so there are no gaps in the indices.

    Returns:
        new_labels: A tensor of shape (n_examples,) with class indices remapped
                    to be contiguous integers starting from 0

        class_map: A tensor of shape (n_classes,) giving the original class
                   labels
    """
    class_map, new_labels = labels.unique(return_inverse=True, sorted=True)
    return new_labels, class_map


def calculate_class_centroids(h, labels):
    classes = labels.unique(sorted=True)
    assert classes[0] == 0, "Labels must start from 0"
    assert classes[-1] == len(classes) - 1, "Labels must be contiguous"
    class_centroids = [h[labels == cls].mean(dim=0) for cls in classes]
    return torch.stack(class_centroids)


@torch.no_grad()
def predict(model: nn.Sequential, x, y_true, skip_layers: int = 1):
    """
    Predict by finding the class with closest centroid to each example.

    TODO: If there's only one or a few examples of a class, the centroid will be
    very close to the example itself, a data leakage issue. This can be fixed by
    excluding the example from the centroid calculation, using the training
    centroids, or somehow using a different method to predict. If there's no
    examples then it won't even be considered, another data leakage issue.
    """
    y_true, class_map = remap_class_labels(y_true)
    assert skip_layers >= 0 and skip_layers < len(model), "Invalid skip_layers"

    distance_matrices: list[torch.tensor] = []
    for h in LayerOutputs(model, x):
        class_centroids = calculate_class_centroids(h, y_true)
        distance_matrices.append(calculate_distance_matrix(h, class_centroids))
    distance_matrix = sum(distance_matrices[skip_layers:])
    predictions = distance_matrix.argmin(1)
    return class_map[predictions]


def centroid_loss(h, y_true, temperature=4.0, regulariser=0.1):
    """
    Loss function based on (squared) distance to the true centroid vs other
    centroids.

    Achieves an error rate of ~1.6%.

    The regulariser is the Label Smoothing Regulariser [1], a float between 0
    and 1. It mixes in a uniform distribution over the class probabilities,
    preventing the model becoming overconfident in its predictions.

    In the context of centroid loss, it can be thought of as forcing the model
    to give some consideration to the off-diagonal elements of the distance
    matrix. We want the outputs to not only be close to their true centroid, but
    also far from the other centroids.

    [1] https://arxiv.org/pdf/1512.00567
    """
    # Distance from h to centroids of each class
    y_true, _ = remap_class_labels(y_true)
    centroids = calculate_class_centroids(h, y_true)
    d = calculate_distance_matrix(h, centroids)

    # Softmax then calculate the cross-entropy loss
    return F.cross_entropy(
        -d * math.exp(temperature),
        y_true,
        reduction="mean",
        label_smoothing=regulariser,
    )


# %%
# Define the model
#
# Must be an iterable of layers.
n_units = 500  # 2000 improves error rate
model = nn.Sequential(
    nn.Sequential(
        Flatten(),
        nn.Linear(784, n_units),
        #nn.BatchNorm1d(n_units),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(n_units, n_units),
        #nn.BatchNorm1d(n_units),
        nn.ReLU(),
    ),
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
learning_rate = 0.001
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 50
train_batch_size = 512
test_batch_size = len(mnist.test_x) # calc centroids over full batch
train_loader = DataLoader(
    list(zip(mnist.train_x, mnist.train_y)), batch_size=train_batch_size, shuffle=True
)
test_loader = DataLoader(
    list(zip(mnist.test_x, mnist.test_y)), batch_size=test_batch_size, shuffle=True
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
            loss = centroid_loss(h, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                x = layer(x)

    # Evaluate the model on the training and test set
    print(
        "[{:>4d}] Training: {:.2%}, Test: {:.2%}".format(
            epoch,
            error_rate(model, train_loader),
            error_rate(model, test_loader),
        )
    )

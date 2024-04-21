# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
from torch.optim import Adam
from torch.utils.data import DataLoader

import mnist
from ff_utils import LayerOutputs, SkipConnection, UnitLength, goodness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def superimpose_label(x, y):
    x = x.clone()
    x[:, 0, 0, :10] = 0
    x[range(x.shape[0]), 0, 0, y] = x.max()
    return x


@torch.no_grad()
def goodness_per_class(model: nn.Sequential, x, skip_layers=0):
    """
    Calculates the goodness for each class label.

    This is the sum of goodness across all layers.

    Returns a tensor [n_examples, n_classes].
    """
    g_per_label = []
    for label in range(10):
        x_candidate = superimpose_label(x, label)
        g = 0
        for i, h in enumerate(LayerOutputs(model, x_candidate)):
            if i < skip_layers:
                continue
            g += goodness(h)
        g_per_label.append(g.unsqueeze(1))  # type: ignore
    return torch.cat(g_per_label, 1)


@torch.no_grad()
def predict(model: nn.Sequential, x, skip_layers=0):
    """Predict the class with highest goodness."""
    return goodness_per_class(model, x, skip_layers=skip_layers).argmax(1)


def make_examples(model: nn.Sequential, x, y_true, epsilon=1e-12):
    """
    Make some positive and negative examples.

    The positive examples are superimposed with their true label. The negative
    examples have a "hard" label with high goodness, excluding the true label.
    """

    # Calculate goodness for each class label
    g = goodness_per_class(model, x)

    # Use the goodness as a probability distribution over all class labels.
    # First, set true label probabilities to zero, then square root to make the
    # distribution less peaked.
    g[range(x.shape[0]), y_true] = 0
    y_hard = torch.multinomial(torch.sqrt(g) + epsilon, 1).squeeze(1)

    x_pos = superimpose_label(x, y_true)
    x_neg = superimpose_label(x, y_hard)
    return x_pos, x_neg


def hinton_loss(h_pos, h_neg, theta=2.0, alpha=1.0):
    """
    Calculate Hinton's Loss as per: https://arxiv.org/pdf/2212.13345.pdf

    Converges very slowly. See SymBa paper for explanation:
    https://arxiv.org/pdf/2303.08418.pdf

    Achieves an error rate on MNIST of ~2.7% for a network with two hidden
    layers of 500 units, after 600 epochs.

    The paper is actually somewhat ambiguous about the loss function, however it
    appears to be log(sigmoid(-x)) where x is (g_pos - theta) or (theta -
    g_neg), based on equations (1) and (3). This is equivalent to the softplus
    function, log(1 + exp(x)).

    Parameters:
        theta (float): A threshold used for both positive and negative examples.
        alpha (float): A scaling factor, also used for both. Hinton sets to 1.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    # Mean of all examples, so not a pairwise comparison:
    loss_pos = F.softplus(theta - g_pos, alpha).mean()
    loss_neg = F.softplus(g_neg - theta, alpha).mean()
    return loss_pos + loss_neg


def symba_loss(h_pos, h_neg, alpha=4.0):
    """
    Calculate SymBa Loss as per: https://arxiv.org/pdf/2303.08418.pdf

    Achieves an error rate on MNIST of ~2.2% for a network with two hidden
    layers of 500 units, after 60 epochs.

    Parameters:
        alpha (float): A scaling factor used for both positive and negative
        examples.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    Delta = g_pos - g_neg
    return F.softplus(-alpha * Delta).mean()


def swish_loss(h_pos, h_neg, alpha=6.0):
    """
    Calculate the Swish variant of SymBa Loss, see README.md for details.

    Achieves an error rate on MNIST of ~1.7% for a network with two hidden
    layers of 500 units, after 60 epochs.

    Parameters:
        alpha (float): A scaling factor used for both positive and negative
        examples.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    Delta = g_pos - g_neg
    return F.silu(-alpha * Delta).mean()


# %%
# Define the model
#
# Must be an iterable of layers, each of which start with a UnitLength()
# sub-layer, to "conceal" goodness from the next layer.

n_units = 500  # 2000 improves error rate
model = nn.Sequential(
    SkipConnection(
        nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    ),
    # nn.Sequential(
    #     UnitLength(),
    #     nn.Conv2d(10, 10, kernel_size=5, padding=2),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    # ),
    nn.Sequential(Flatten(), UnitLength(), nn.Linear(784, n_units), nn.ReLU()),
    nn.Sequential(UnitLength(), nn.Linear(n_units, n_units), nn.ReLU()),
).to(device)


def error_rate(model: nn.Sequential, data_loader: DataLoader) -> float:
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        predicted = predict(model, x, skip_layers=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return 1 - correct / total


# %%
# Train the model
torch.manual_seed(42)
loss_fn = swish_loss  # hinton_loss
learning_rate = 0.1 if loss_fn is hinton_loss else 0.35
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 600 if loss_fn is hinton_loss else 60
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

        # Positive examples: the true label
        # Negative examples: a "hard" label that is not the true label
        # TODO: we could move the negative example generation inside the layer loop
        x_pos, x_neg = make_examples(model, x, y)

        # Train layers in turn, using backpropagation locally only
        model.train()
        for layer in model:
            h_pos, h_neg = layer(x_pos), layer(x_neg)
            loss = loss_fn(h_pos, h_neg)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                x_pos, x_neg = layer(x_pos), layer(x_neg)

    # Evaluate the model on the training and test set
    if epoch % 5 == 0:
        print(
            "[{:>4d}] Training: {:.2%}, Test: {:.2%}".format(
                epoch,
                error_rate(model, train_loader),
                error_rate(model, test_loader),
            )
        )

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import split
from torch.optim import Adam
from utils import LayerOutputs, UnitLength

def superimpose_label(x, y):
    x = x.clone()
    x[:, :10] = 0
    x[range(x.shape[0]), y] = x.max()
    return x

# %%
def goodness(h): 
    """Goodness is the *mean* squared activation of a layer."""
    return h.pow(2).mean(1)

@torch.no_grad()
def goodness_per_class(model, x):
    """
    Calculates the goodness for each class label.

    This is the sum of goodness across all layers.
     
    Returns a tensor [n_examples, n_classes].
    """
    g_per_label = []
    for label in range(10):
        x_candidate = superimpose_label(x, label)
        g_candidate = sum(goodness(h) for h in LayerOutputs(model, x_candidate))
        g_per_label.append(g_candidate.unsqueeze(1)) # type: ignore
    return torch.cat(g_per_label, 1)

@torch.no_grad()
def predict(model, x):
    """Predict the class with highest goodness."""
    return goodness_per_class(model, x).argmax(1)

# %%
def make_examples(model, x, y_true, epsilon=1e-12):
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

# %%
def hinton_loss(h_pos, h_neg, theta=2.0):
    """
    Calculate Hinton's Loss as per: https://arxiv.org/pdf/2212.13345.pdf

    Converges very slowly. See SymBa paper for explanation:
    https://arxiv.org/pdf/2303.08418.pdf

    Achieves an error rate on MNIST of ~2.7% for a network with two hidden
    layers of 500 units, after 600 epochs.

    Paramaters:
        theta (float): A threshold used for both positive and negative examples.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    return F.softplus(theta - g_pos).mean() + F.softplus(g_neg - theta).mean()

# %%
def symba_loss(h_pos, h_neg, alpha=4.0):
    """
    Calculate SymBa Loss as per: https://arxiv.org/pdf/2303.08418.pdf

    Achieves an error rate on MNIST of ~2.2% for a network with two hidden
    layers of 500 units, after 60 epochs.

    Paramaters:
        alpha (float): A scaling factor used for both positive and negative
        examples.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    Delta = g_pos - g_neg
    return F.softplus(-alpha*Delta).mean()

# %%
def triplet_loss(h_pos, h_neg, margin=0.5):
    """
    Calculates the Triplet Loss.
    
    Adapted from the standard loss for Siamese Networks:
    https://en.wikipedia.org/wiki/Triplet_loss

    >>> Loss_i = max(||neg - anchor|| - ||pos - anchor|| + margin, 0)

    We use goodness as the distance measure (the squared Euclidean distance),
    where the "anchor point" is the origin (i.e. zero goodness). We therefore
    want negative inputs to be near the anchor, and positive inputs to be far
    from the anchor. Note, the anchor in the standard formulation is for the
    positive input, but we use the anchor for the negative input here, so the
    roles of negative and positive are reversed. 

    Achieves an error rate of ~2.1% for a network with two hidden layers of 500
    units, after 60 epochs.
    """
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    return F.relu(g_neg - g_pos + margin).mean()

# %%
def smoothed_triplet_loss(h_pos, h_neg, beta=5.0):
    """       
    Calculate the Smoothed Triplet Loss.
    
    The Swish activation function (aka SiLU) acts like a smoothed ReLU. It bakes
    in a margin of approx 2.6/beta, and includes a smooth valley around the
    margin point, which helps with convergence, and allows us to increase the
    learning rate. 
    
    Achieves an error rate of ~1.65%, after 60 epochs.

    The value of beta determines the margin point. It also affects the
    sharpness of the curvature around the margin, so an additional offset
    parameter may be required for more complex problems.
    """

    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    return F.softplus(beta * (g_neg - g_pos)).mean() #F.silu(beta * (g_neg - g_pos)).mean()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# The data is pre-processed, to speed up this script
x_tr = torch.load('./data/MNIST/baked/train_x.pt', device)
y_tr = torch.load('./data/MNIST/baked/train_y.pt', device)
x_te = torch.load('./data/MNIST/baked/test_x.pt', device)
y_te = torch.load('./data/MNIST/baked/test_y.pt', device)

# %% Define the model
# ----------------
# Must be an iterable of layers, each of which start with a UnitLength()
# sub-layer, to "conceal" goodness from the next layer.
n_units = 500 # 2000 improves error rate
model = nn.Sequential(
    nn.Sequential(UnitLength(), nn.Linear(784, n_units), nn.ReLU()),
    nn.Sequential(UnitLength(), nn.Linear(n_units, n_units), nn.ReLU()),
).to(device)

# %%
# Evaluate the model on the training and test set
def print_evaluation(epoch=None):
    global model, x_tr, y_tr, x_te, y_te
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x), y)
    train_error = prediction_error(x_tr, y_tr)
    test_error = prediction_error(x_te, y_te)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"
    print(f"[{epoch_str}] Training: {train_error*100:>5.2f}%\tTest: {test_error*100:>5.2f}%")

# %%
# Training parameters
torch.manual_seed(42)
loss_fn = smoothed_triplet_loss
learning_rate = 0.1 if loss_fn is hinton_loss else 0.35
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 1 + (600 if loss_fn is hinton_loss else 60)
batch_size = 4096

# %%
# Train the model
print_evaluation()
for epoch in range(num_epochs):

    # Mini-batch training
    for x, y in zip(split(x_tr, batch_size), split(y_tr, batch_size)):

        # Positive examples: the true label
        # Negative examples: a "hard" label that is not the true label
        # TODO: we could move the negative example generation inside the layer loop
        x_pos, x_neg = make_examples(model, x, y)

        # Train layers in turn, using backprop locally only
        for layer in model:
            h_pos, h_neg = layer(x_pos), layer(x_neg)
            loss = loss_fn(h_pos, h_neg)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            x_pos, x_neg = h_pos.detach(), h_neg.detach()

    # Evaluate the model on the training and test set
    if (epoch + 1) % 5 == 1:
        print_evaluation(epoch)

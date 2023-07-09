# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import split
from torch.optim import Adam
from utils import LayerOutputs, UnitLength

# %%
def distance_to_centroids(h, y_true, epsilon=1e-12):
    """
    Calculates the mean squared distance to the centroid of each class. 

    Returns a tensor of shape [n_examples, 10].
    """
    safe_mean = lambda x, dim: x.sum(dim) / (x.shape[dim] + epsilon)
    # TODO: what if class is missing? 
    # * determine centroids only for classes that are present, and return torch.unique(y_true)
    # * or treat centroids as trainable parameters, so they slowly update
    class_centroids = torch.stack([safe_mean(h[y_true == i],0) for i in range(10)], dim=1) # [n_in, 10]
    x_to_centroids = h.unsqueeze(2) - class_centroids # [n_examples, n_in, 10]
    return x_to_centroids.pow(2).mean(1) # [n_examples, 10]
    
@torch.no_grad()
def predict(model, x, y_true, skip_layers=1):
    """Predict by finding the class with closest centroid to each example."""
    d = sum([distance_to_centroids(h, y_true) for h in LayerOutputs(model, x)][skip_layers:])
    return d.argmin(1) # type: ignore

# %%
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
    d2_true = d2[range(d2.shape[0]), y_true] # ||anchor - positive||^2
    d2_near = d2[range(d2.shape[0]), y_near] # ||anchor - negative||^2
    return F.silu(alpha * (d2_true - d2_near)).mean(), (y_true == y_near).float().sum().item(), d2_true.mean(), d2_near.mean()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# The data is pre-processed, to speed up this script
x_tr = torch.load('./data/MNIST/preprocessed/train_x.pt', device)
y_tr = torch.load('./data/MNIST/preprocessed/train_y.pt', device)
x_te = torch.load('./data/MNIST/preprocessed/test_x.pt', device)
y_te = torch.load('./data/MNIST/preprocessed/test_y.pt', device)

# %% 
# Define the model
# ----------------
# Must be an iterable of layers. I find it works best if each layer starts with
# a UnitLength() sub-layer.
n_units = 500 # 2000 improves error rate
model = nn.Sequential(
    nn.Sequential(UnitLength(), nn.Linear(784, n_units), nn.ReLU()),
    nn.Sequential(UnitLength(), nn.Linear(n_units, n_units), nn.ReLU()),
    #nn.Sequential(UnitLength(), nn.Linear(n_units, 3)),
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
learning_rate = 0.05
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 120+1
batch_size = 4096

# %%
# Train the model
print_evaluation()
for epoch in range(num_epochs):

    # Mini-batch training
    n_same = [0.0,0.0,0.0]
    n_samples = [0.0,0.0,0.0]
    total_d2_true = [0,0,0]
    total_d2_near = [0,0,0]
    for x, y in zip(split(x_tr, batch_size), split(y_tr, batch_size)):

        # Train layers in turn on same mini-batch, using backpropagation locally only
        for i,layer in enumerate(model):
            h = layer(x)
            temperature = 4
            loss, same, d2_true, d2_near = centroid_loss(h, y, temperature=temperature)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            n_same[i] += same
            n_samples[i] += x.shape[0]
            total_d2_true[i] += d2_true
            total_d2_near[i] += d2_near
            with torch.no_grad():
                x = layer(x)

    # Evaluate the model on the training and test set
    if (epoch + 1) % 5 == 1:
        print_evaluation(epoch)
        for i,_ in enumerate(model):
            print(f"Layer {i}: {n_same[i]/n_samples[i]} same, {total_d2_near[i]/total_d2_true[i]} ratio")
    
# %%
# Visualise the model, last layer only
#%matplotlib qt 
import numpy as np
import matplotlib.pyplot as plt
def plot_model(model, x, y, title=""):
    with torch.no_grad():
        h = model(x)
    n_samples = 10000
    h = h[:n_samples].cpu().numpy()
    y = y[:n_samples].cpu().numpy()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors
    for class_id in range(10):
        ax.scatter(h[y == class_id, 0], h[y == class_id, 1], h[y == class_id, 2], 
                   color=colors[class_id], cmap='tab10',
                   s=3, alpha=0.8)

    legend = plt.legend(handles=ax.collections, labels=range(10), loc='upper right')
    for handle in legend.legend_handles:
        handle._sizes = [20]
    plt.title(title)
    plt.show()
plot_model(model, x_tr, y_tr, "Training set")
# %%

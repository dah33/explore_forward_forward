# Explore Forward-Forward Algorithm

In this repo, I explore Geoffrey Hinton's [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) for layer-wise training of neutral networks without backpropagation. His new learning procedure replaces the forward and backward passes of backpropagation by two forward passes, one with positive (i.e. real) data and the other with negative data.

I present a reference implementation of Hinton's algorithm, and two alternative loss functions that achieve a lower error rate on the MNIST dataset. I also propose a new algorithm that uses the centroids of each label.

## Reference Implementation

In [hinton_goodness.py](.\hinton_goodness.py) I implement Hinton's algorithm using Pytorch on the MNIST dataset using a simple MLP with two hidden layers of 500 rectified linear units, as presented in his paper. 

Hinton's proposed loss function for the output of layer $k$ is:

$$
\mathcal{L}^{(k)}(x) = \log(1+ e^{y(G - \theta)})
$$

where: 

* $G = ||f(x)||^2$ is the *goodness* measure, where $f(x)$ is the output of layer $k$, parameterised by $x$, the input to *first* layer.
* $y$ is the class of the input: 
  - $+1$ for *positive* inputs, with the correct label superimposted;
  - $-1$ for *negative* inputs, with an incorrect label superimposed.
* $\theta$ is some threshold, a hyperparameter, the same for all layers.

The loss for the first layer is summed over all positive and negative inputs, then minimised, before the output of the now trained layer is passed to the next layer, and so on.

The network makes a prediction by trying all possible labels with an input, and choosing the one with the highest goodness.

Like Hinton, I test this algorithm on the MNIST dataset. I show that training a simple model with two hidden layers of 500 units converges slowly to an error rate of ~2.7%, after 600 epochs. 

### Negative Examples

Hinton leaves open the question of the best source of negative examples. I mine for "hard negative" examples, by choosing an incorrect label with a high goodness. This is achieved by treating the goodness as a probability distribution over the incorrect labels, and sampling from it.

## Alternative Loss Functions

### SymBa Loss

This [paper](https://arxiv.org/pdf/2303.08418.pdf) proposes a loss function that works with *pairs* of positive and negative inputs:

$$
\begin{align}
\mathcal{L}^{(k)}(\mathcal{P},\mathcal{N}) &= \log(1+ e^{-\alpha\Delta}) \\
    &= \text{softplus}(-\alpha\Delta)
\end{align}
$$

where:

* $\Delta = G_\mathcal{P} - G_\mathcal{N}$ is the difference in goodness for the layer of the positive and negative inputs, $\mathcal{P}$ and $\mathcal{N}$.
* $\alpha$ is a scaling factor, a hyperparameter, the same for all layers.

The paper also demonstrates why training using this loss function converges more quickly than Hinton's proposed loss function.

I show that training the same network as before, but using the SymBa loss function, converges more quickly to MNIST error rate of ~2.2%, after 60 epochs. 

### Triplet Loss

I reformulate the SymBa loss, by drawing a parallel to the [Triplet Loss](https://en.wikipedia.org/wiki/Triplet_loss) function used in Siamese Networks:

$$
\mathcal{L}^{(k)}(\mathcal{A},\mathcal{P},\mathcal{N}) = 
    \text{max}(||f(\mathcal{A}) - f(\mathcal{P})||^2 - ||f(\mathcal{A}) - f(\mathcal{N})||^2 + \alpha, 0)
$$

where: 

* $\mathcal{A}$ is the *anchor* input
* $\mathcal{P}$ is a *positive* input from the same class as the anchor
* $\mathcal{N}$ is a *negative* input from a different class to the anchor
* $\alpha$ is the margin, a hyperparameter, the same for all layers.

I define two classes, "correct" and "incorrect", so that $\mathcal{P}$ is an input with the correct label superimposed, and $\mathcal{N}$ is the same input, but with an incorrect label superimposed. The anchor input $\mathcal{A}$ is the origin (i.e. zero goodness) with a class of "correct". I thus want the negative input to be near the anchor, and the positive input to be far from the anchor, where the distance measure is goodness, the squared Euclidean distance. This reverses the roles of positive and negative in the above formulation. This leads to the loss function:

$$
\begin{align}
\mathcal{L}^{(k)}(\mathcal{P},\mathcal{N}) &= 
    \text{max}\big(G_\mathcal{N} - G_\mathcal{P} + \alpha, 0\big) \\
    &\approx \text{SiLU}(-\beta\Delta)
\end{align}
$$

where $\beta \approx {2.6}/{\alpha}$, and $\text{SiLU}(x) = x\sigma(x)$ is the [Sigmoid Linear Unit](https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions) activation function.

I show that this formulation improves on the SymBa loss, reducing the MNIST error rate to ~1.65%. After just one epoch, the error rate is ~12%. Increasing the number of units in both hidden layers to 2000, reduces the error rate to ~1.35%.

# Centroid Algorithm

In `centroid.py` propose a new algorithm that sets $\mathcal{N}$ and $\mathcal{P}$ to be the "centroids" of each class. 

The motivation for the algorithm is the observation that my implementation of Hinton's algorithm requires multiple forward passes for each example, to:

1. Mine for a hard negative input by calculating the goodness of the example with each incorrect label superimposed, which requires a forward pass for each incorrect label.
2. Layer-wise training, with forward passes for the example and chosen negative input.

Prediction also requires a forward pass for each possible label.

The centroid algorithm requires only one forward pass for each example. The mean output for a layer across all examples in a class is called the centroid. The centroid of the correct class for an example is the positive input, and the centroid of a "nearby" incorrect class is the negative input. The loss function is the same as for the Triplet Loss.

The network makes a prediction by looking for the class with the closest centroids, summing over all layers. This could be achieved efficiently using a modern vector database.

I show that this algorithm achieves an error rate on MNIST of ~2.0% after 120 epochs. After just one epoch, the error rate is ~11%.

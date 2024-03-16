# Explore Forward-Forward Algorithm

In this repo, I explore Geoffrey Hinton's [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) for layer-wise training of neutral networks without backpropagation. His new learning procedure replaces the forward and backward passes of backpropagation by two forward passes, one with positive (i.e. real) data and the other with negative data.

I present a reference implementation of Hinton's algorithm, and two alternative loss functions that achieve a lower error rate on the MNIST dataset. I also propose a new algorithm that uses the centroids of each label.

## Reference Implementation

In [ff_hinton.py](.\ff_hinton.py) I implement Hinton's algorithm using PyTorch on the MNIST dataset using a simple MLP with two hidden layers of 500 rectified linear units, as presented in his paper. 

Hinton's proposed loss function (see equations (1) and (3) in the paper) for the output of layer $k$:

$$
\mathcal{L}(x) = \log(1+ e^{y(\theta - G_x)})
$$

where: 

* $G_x = ||f(x)||^2$ is the *goodness* measure, where $f(x)$ is the output of layer $k$, parameterised by $x$, the *input* to the network. In other words, an embedding of $x$.
* $y$ is the class of the input, $x$: 
  - $+1$ for *positive* inputs, with the correct label superimposed;
  - $-1$ for *negative* inputs, with an incorrect label superimposed.
* $\theta$ is some threshold, a hyperparameter, the same for all layers.

The loss for the first layer is summed over all positive and negative inputs, then minimised, before the output of the now trained layer is passed to the next layer, and so on.
  
The network makes a prediction by superimposing all possible labels onto the input, and choosing the variant with the highest total goodness across the layers. I choose to average the goodness per neuron in each the layer, then sum across all layers. This ensures the $\theta$ are comparable, and all layers are weighted equally, regardless of their size. Hinton chooses to discard the goodness of the first layer for deeper networks.

Like Hinton, I test this algorithm on the MNIST dataset. I show that training a simple model with two hidden layers of 500 units converges slowly to an error rate of ~2.7%, after 600 epochs. 

### Negative Examples

Hinton leaves open the question of the best source of negative examples. I mine for "hard negative" examples, by choosing an incorrect label with a high goodness. This is achieved by treating the goodness as a probability distribution over the incorrect labels, and sampling from it.

## Alternative Loss Functions

### SymBa Loss

This [paper](https://arxiv.org/pdf/2303.08418.pdf) proposes a loss function that works with *pairs* of positive and negative inputs:

$$
\mathcal{L}(\mathcal{P},\mathcal{N}) = \log(1+ e^{-\alpha(G_\mathcal{P} - G_\mathcal{N})})
$$

where $G_\mathcal{P} - G_\mathcal{N}$ is the difference in goodness for the layer for the positive and negative inputs, $\mathcal{P}$ and $\mathcal{N}$, and $\alpha$ is a scaling factor, a hyperparameter, the same for all layers.

The paper explains why training using this loss function converges more quickly than Hinton's proposed loss function.

I show that training the same network as before, but using the SymBa loss function, converges more quickly to an error rate of ~2.2%, after 60 epochs. 

### Swish Variant

The SymBa loss uses $log(1 + e^x)$ which is a soft approximation to $\text{max}(0, x)$. Another soft approximation is the [Swish](https://en.wikipedia.org/wiki/Swish_functions) function $x\sigma(x)$. It is non-monotonic below zero, which likely has a regularising effect. The SymBa loss becomes:

$$
\mathcal{L}(\mathcal{P},\mathcal{N}) = \frac{-\alpha(G_\mathcal{P} - G_\mathcal{N})}{1 + e^{-\alpha(G_\mathcal{P} - G_\mathcal{N})}}
$$

I show that this formulation improves on the SymBa loss, reducing the error rate to ~1.65%. After just one epoch, the error rate is ~12%. 

Increasing the number of units in both hidden layers to 2,000, as per Hinton's paper, reduces the error rate to ~1.35% using the Swish Variant.

## Alternative Implementation

See `ff_pytorch_example.py` for an implementation based on the [Pytorch
Example](https://github.com/pytorch/examples/blob/main/mnist_forward_forward/main.py).

# Centroid Algorithm

In [`ff_centroids.py`](./ff_centroids.py) propose a new algorithm that sets anchors as the "centroids" of each class. 

The motivation for the algorithm is the observation that my implementation of Hinton's algorithm requires multiple forward passes for each example, to:

1. Mine for a hard negative input by calculating the goodness of the example with each incorrect label superimposed, which requires a forward pass for each incorrect label.
2. Layer-wise training, with forward passes for the example and chosen negative input.

Prediction also requires a forward pass for each possible label.

The centroid algorithm requires only one forward pass for each example. The mean output for a layer across all examples in a class is called the centroid. The centroid of the correct class for an example is the positive input, and the centroid of a "nearby" incorrect class is the negative input. The loss function is the same as for the Triplet Loss.

The network makes a prediction by looking for the class with the closest centroids, summing over all layers. This could be achieved efficiently using a modern vector database.

I show that this algorithm achieves an error rate on MNIST of ~1.70% after 60 epochs. After just one epoch, the error rate is ~10%.

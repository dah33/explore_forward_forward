# Explore Forward-Forward Algorithm

In this repo, I explore Geoffrey Hinton's [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) for layer-wise training of neutral networks without backpropagation. His new learning procedure replaces the forward and backward passes of backpropagation by two forward passes, one with positive (i.e. real) data and the other with negative data.

I present a reference implementation of Hinton's algorithm, and two alternative loss functions that achieve a lower error rate on the MNIST dataset. I also propose a new algorithm that sets the negative and positive inputs to be the centroids of each class.

## Reference Implementation

In `goodness.py` I implement Hinton's algorithm using Pytorch on the MNIST dataset using a simple MLP with two hidden layers of 500 rectified linear units, as presented in his paper. For training the model, I first implement Hinton's proposed loss function:

$$
\mathcal{L}(\mathcal{P},\mathcal{N}) = 
    \log(1+ e^{G_{\mathcal{P}} - \theta}) + \log(1+ e^{\theta - G_{\mathcal{N}}})
$$

where: 

* $\mathcal{P}$ is a *positive input*: an example with the correct label superimposed
* $\mathcal{N}$ is a *negative input*: an example with an incorrect label superimposed
* $f$ is the embedding: the output of a given layer of the neural network, parameterised by the input given to the *first* layer.
* $\mathcal{G}_{\mathcal{P}} = \sum_i f(\mathcal{P})_i^2$ is the goodness measure for the positive input, and similarly for the negative input
* $\theta$ is some threshold, a hyperparameter

The training is layer-wise. Given the pair of inputs, the loss for a first layer is summed over all examples, then minimised, before the output of the now trained layer is passed to the next layer, and so on.

The network makes a prediction by trying all possible labels with an input, and choosing the one with the highest goodness.

I show that training using this loss function converges slowly to error rate on the MNIST test set of ~2.7%, after 600 epochs. 

### Negative Examples

Hinton leaves open the question of the best source of negative examples. I mine for "hard negative" examples, by choosing the incorrect label with a high goodness. This is achieved by treating the goodness as a probability distribution over the incorrect labels, and sampling from it.

## Triplet Loss

I also demonstrate two more loss functions, which converge quicker to a lower error rate. These are based on the [Triplet Loss](https://en.wikipedia.org/wiki/Triplet_loss) function used in Siamese Networks:

$$
\mathcal{L}(\mathcal{A},\mathcal{P},\mathcal{N}) = 
    \text{max}(||f(\mathcal{A}) - f(\mathcal{P})||_2 - ||f(\mathcal{A}) - f(\mathcal{N})||_2 + \alpha, 0)
$$

where: 

* $\mathcal{A}$ is the *anchor input*
* $\mathcal{P}$ is a *positive input* of the same class as $\mathcal{A}$
* $\mathcal{N}$ is a *negative input* from a different class from $\mathcal{A}$
* $f$ and $\mathcal{G}$ are as above
* $\alpha$ is the margin, a hyperparameter
* $||\cdot||_2$ is the Euclidean distance

There are two classes, "true" and "false", meaning the correct or incorrect label is superimposed on the example. I use goodness (sum of squared activities) in place of the Euclidean distance measure (although that works as well). The anchor input is the origin (i.e. zero goodness) with a class of "false". I thus want the negative (i.e. false) input to be near the anchor, and positive (i.e. true) input to be far from the anchor. This reverses the roles of positive and negative in the above formulation. This leads to the loss function:

$$
\mathcal{L}(\mathcal{P},\mathcal{N}) = 
    \text{max}(G_{\mathcal{N}} - G_{\mathcal{P}} + \alpha, 0)
$$

I show that training for 60 epochs, using a smoothed version of this loss function, achieves an error rate on MNIST of ~1.7%. After just one epoch, the error rate is ~12%.

## Centroid Algorithm

In `centroid.py` propose a new algorithm that sets $\mathcal{N}$ and $\mathcal{P}$ to be the "centroids" of each class. 

The motivation for the algorithm is the observation that my implementation of Hinton's algorithm requires multiple forward passes for each example, to:

1. Mine for a hard negative input by calculating the goodness of the example with each incorrect label superimposed, which requires a forward pass for each incorrect label.
2. Layer-wise training, with forward passes for the example and chosen negative input.

Prediction also requires a forward pass for each possible label.

The centroid algorithm requires only one forward pass for each example. The mean output for a layer across all examples in a class is called the centroid. The centroid of the correct class for an example is the positive input, and the centroid of a "nearby" incorrect class is the negative input. The loss function is the same as for the Triplet Loss.

The network makes a prediction by looking for the class with the closest centroids, summing over all layers. This could be achieved efficiently using a modern vector database.

I show that this algorithm achieves an error rate on MNIST of ~2.0% after 120 epochs. After just one epoch, the error rate is ~11%.

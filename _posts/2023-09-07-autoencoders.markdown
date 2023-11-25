---
layout: post
comments: true
title:  "Notes on Autoencoders"
excerpt: "brief, an autoencoder is a neural network that attempts to “copy” its input to its output."
date:   2023-09-07 22:00:00
category: "Architectures"
mathjax: true
---

# Autoencoders

In brief, an autoencoder is a neural network that attempts to “copy” its input to its output.

The network consists of two parts: an **encoder** and a **decoder**, the encoder translates input into an intermediate state. The decoder takes this intermediate state and tries to output something that is approximates the input.

Encoder: \\( \boldsymbol{h} = f(\boldsymbol{x}) \\)

Decoder: \\( \boldsymbol{r} = g(\boldsymbol{h} \\)

Autoencoders are designed to be unable to copy the input perfectly. Therefore, the model is forced to prioritize which aspects should be copied and which should not. It is this property that makes autoencoders cool!

The different flavours of autoencoders is all about how we constrain the architecture so as to do this “imperfect” copying of the inputs to the outputs. 

We use two terms: `code` and `capacity`

`code` refers to \\( \boldsymbol{h} \\) (i.e. the intermediate state of the inputs)

`capacity` refers to the model’s ability to fit various functions. Higher capacity means it can fit more complex functions. Capacity could be the number and types of layers in the encoder/decoder, what activation functions are used, what loss function we are optimizing over, the batch size, the number of neurons per layer, what regularization technique we are using, etc.

## Undercomplete Autoencoders

One traditional way to achieve the incomplete copying is by encoding the inputs to a lower dimensional subspace. This is what we mean by `undercomplete`. 

Learning this network is done by minimizing the following loss function:

\\( L(\boldsymbol{x}, g(f(\boldsymbol{x})) \\), where \\( L \\) is a loss function penalizing the prediction for being dissimilar from the input (MAE for example).

When the decoder is linear and the loss function is MAE, what we get is the same as PCA.

Why? because, PCA is about projecting the information onto vectors of a lower dimensional subspace, i.e. linear directions! 

How? 

```python
enc = nn.Linear(30000, 1500)
dec = nn.Linear(1500, 30000)

x = enc(x)
x = dec(x)
```

We take input, compress it to 1500 

Take the 1500 and expand it to the original size

PCA aims at reducing the error, hence the loss function MAE is the appropriate objective for the autoencoder. Minimizing this loss is equivalent to maximizing the variance of the projected data, which is the objective of PCA.

If we apply a non-linearity between the two layers, we can thus learn non-linear relations of the data which is more powerful in many cases.

## Regularized autoencoders

Since autoencoding is about finding meaningful representations of the data, the data’s complexity should determine the dimensionality of the hidden code.

One should be able to choosing the code dimension and the capacity based on the complexity of the distribution to be modelled. 

Hence, we should theoretically be able to have a hidden dimensionality far greater than the input dimensionality if we want to without suffering from overfitting due to the model being `overcomplete` (i.e. having greater dimensionality in the hidden state).

This is where regularization comes in. 

By using a loss function that encourages the model to have other properties besides being able to copy its input to its output. Properties such as “sparsity of representation”, “robustness to noise or missing inputs”, smallness of the derivative of the representation

**Sparsity of representation**

Here we penalize the model for having non-zero values in the latent space.

Thus, the model naturally tries to find the “best” neurons to activate and let others die out.

Sparsity loss = \\( \sum_{i=1}^n \|z_i\|_1 \\)

Here the z_i is the ith element in the latent space, and the norm is the L1 norm


**Smallness of the derivative of the representation**

This encourages the latent space to change slowly / smoothly with respect to changes in the input.

 Why do we want this? 

- to increase stability and generalization. If the model changes rapidly depending on the input, the model won’t learn as much because it is too sensitive to noise.
- Clustering
	- by encouraging a smooth latent space, similar inputs will have similar codes and is therefore easier to cluster.
- Interpretability
	- A smooth latent space is often easier to interpret. If the latent variables change smoothly as a function of the input, it's easier to understand what each latent variable is capturing. This is particularly important in fields like healthcare, finance, or any domain where understanding the model's behavior is crucial.

This is often implemented using a contractive penalty on the loss function, which discourages large derivatives in the latent space.

Contractive loss =  \\( \lambda \sum_{i=1}^n(\frac{\partial z_i}{\partial x})^2 \\)

the lambda is the parameter that determines the strength of the penalty.

The book describes the following loss function: 

$$
\begin{align}
L(\boldsymbol{x}, g(f(\boldsymbol{x}))) + \Omega(\boldsymbol{h}, \boldsymbol{x})
\end{align}
$$

Where \\( \Omega \\) is the regularization term and is defined as follows:

$$
\begin{align}
\Omega(\boldsymbol{h}, \boldsymbol{x}) =\ \lambda\sum_i \|\nabla \boldsymbol{x}h_i\|^2 .
\end{align}
$$

This forces the model to learn a function that does not change much when **x** changes slightly. Because this penalty is applied only at training examples, it forces the autoencoder to learn features that capture information about the training distribution. An autoencoder regularized in this way is called a `contractive autoencoder` or CAE.

**Robustness to Noise or to Missing inputs**

Robustness means that the model is able to reconstruct input even if noice has been introduced.

Denoising autoencoders (DAE) achieve this by training on noisy versions of the input but using the clean version for computing the loss.

If \\(x\\) is the clean input and \\(x^\prime\\) is the noisy input, the reconstruction loss is:

Reconstruction loss = \\(\|x - g(f(x^\prime)\|^2 = \sum_{i=1}^n (x_i - \hat{x}_i)^2\\), where the norm is the L2 norm

### Sparse autoencoders

These are autoencoders whose training criterion involves a sparsity penalty \\(\Omega(\boldsymbol{h})\\) on the code layer in addition to the reconstruction error:

$$
\begin{align}
L\left(\boldsymbol{x}, g(f(\boldsymbol{x}))\right) + \Omega(\boldsymbol{h})
\end{align}
$$

where \\(\boldsymbol{h} = f(\boldsymbol{x})\\).

An autoencoder that has been regularized to be sparse must respond to unique statistical features of the dataset it has been trained on, rather than simply acting as an identity function. In this way, training to perform the copying task with a sparsity penalty can yield a model that has learned useful features as a byproduct.

We can think of the penalty Ω(h) simply as a regularizer term added to a feedforward network whose primary task is to copy the input to the output

****Relation to bayesian inference****

[Bayesian statistics](../math/statistics/bayesian_stat.md)

Unlike other regularizers such as weight decay, there is not a straightforward Bayesian interpretation to this regularizer. Regularized autoencoders defy such an interpretation because the regularizer depends on the data and is therefore by definition not a prior in the formal sense of the word. We can still think of these regularization terms as implicitly expressing a preference over functions.


💡 Rather than thinking of the sparsity penalty as a regularizer for the copying task, we can think of the Autoencoder framework as approximating maximum likelihood training of a generative model that has latent variables.



The difference:

**Prior on Model Parameters (\\(p(\theta)\\))**

This is the "traditional" prior we often talk about in Bayesian statistics. It represents our initial belief about the model parameters θ before we've seen any data. For example, in a linear regression model, θ could be the coefficients, and p(θ) could be a Gaussian distribution centered at zero, indicating a belief that the coefficients should be small.

**Prior on Latent Variables (pmodel(h))**

This is a different kind of prior. It's not about the model parameters but about the latent (hidden) variables h in the model. The latent variables are variables that are not directly observed but are inferred from the observed variables x.

In the context of a generative model like a Variational Autoencoder or a Gaussian Mixture Model, pmodel(h) represents the model's belief about what kinds of latent variables are likely before it sees any actual data *x*.

Suppose we have a model with visible variables \\(\boldsymbol{x}\\) and latent variables \\(\boldsymbol{h}\\), with an explicit joint distribution 
\\(p_{\text{model} }(\boldsymbol{x},\boldsymbol{h}) = p_{\text{model} }(\boldsymbol{h})p_{\text{model} }(\boldsymbol{x}|\boldsymbol{h})\\) .

Here \\(p_{\text{model} }(\boldsymbol{h})\\) is understood as the model’s prior distribution over the latent variables, representing the model’s beliefs prior to seeing \\(\boldsymbol{x}\\).

The log likelihood can be decomposed as:

$$
\begin{align}
\log p_{\text{model} }(\boldsymbol{x}) = \log \sum_{\boldsymbol{h} } p_{\text{model} }(\boldsymbol{h},\boldsymbol{x})
\end{align}
$$

This equation sums over all possible values of ***h*** to get the total likelihood of ***x.*** This is computationally expensive and often intractable for complex models and large datasets.

The Autoencoder simplifies this by an approximation. We can think of the autoencoder as approximating this sum with a point estimate for just one highly likely value for \\(\boldsymbol{h}\\). This is then used to approximate \\(\log p_{\text{model} }(\boldsymbol{x})\\).

The encoder part of the autoencoder maps ***x*** to a point in the latent space ***h***, and the decoder reconstructs ***x*** from ***h***. This can be seen as a "shortcut" to directly estimate a likely ***h*** without having to sum over all possible ***h***.

From this point of view, with the chosen ***h*** we are maximizing:

$$
\begin{align}
	\log p_{\text{model} }(\boldsymbol{h},\boldsymbol{x}) = \log p_{\text{model} }(\boldsymbol{h}) + \log p_{\text{model} }(\boldsymbol{x}|\boldsymbol{h}).
\end{align}
$$

\\(\log p_{\text{model} }(\boldsymbol{h})\\) can be sparsity-inducing. For example with the Laplace prior:

$$
\begin{align}
p_{\text{model} }(h_i)=\frac{\lambda}{2}e^{-\lambda|h_i|},	
\end{align}
$$

corresponds to the absolute value sparsity penalty. Expressing the log-prior as an absolute value penalty, we obtain:

$$
\begin{align}
\Omega(\boldsymbol{h}) = \lambda\sum_i|h_i|	
\end{align}
$$

$$
\begin{align}
-\log p_{\text{model} }(\boldsymbol{h}) = \sum_i\left(\lambda|h_i|-\log\frac{\lambda}{2}\right) = \Omega(\boldsymbol{h})+\text{const}	
\end{align}
$$

Where the constant term depends only on \\(\lambda\\) and not \\(\boldsymbol{h}\\). Typically, \\(\lambda\\) is treated as a hyperparameter and the constant term is discarded since it does not affect parameter learning.

 

The key insight here is that the sparsity penalty is not an arbitrary regularization term but a direct consequence of the model's prior distribution over its latent variables ***h***. This gives us a different way to think about what an autoencoder is doing:

1. **Generative model:** The autoencoder is approximating a generative model with latent variables **h** and observed variables **x** 


2. **Useful features:** The features learned by the autoencoder are useful because they describe the latent variables ***h*** that explain the input **x**

## Stochastic Encoders and decoders

Stochastic autoencoders introduce randomness into the encoding and decoding processes making them more flexible and generalizable to new data.

**Determinism vs stochastic:**

- In a deterministic autoencoder, the encoder and decoder functions are deterministic mappings. Given an input *x*, the encoder produces a fixed latent representation *h*, and the decoder produces a fixed reconstruction \\(\hat{x}\\).
- In contrast, stochastic encoders and decoders introduce randomness into these mappings. The encoder produces a distribution over latent variables *h* given an input *x*, and the decoder produces a distribution over reconstructed inputs \\(\hat{x}\\) given a latent variable *h*.

If X is real valued we use a gaussian distribution and take the negative log-likelihood yields a mean squared error criterion.

*Real-valued X example:*

Stochastic encoder: 

$$
\begin{align}
p(\boldsymbol{h}|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{h}, \mu(\boldsymbol{x}), \sigma(\boldsymbol{x}))
\end{align}
$$

Stochastic decoder:

$$
\begin{align}
p(\boldsymbol{x} | \boldsymbol{h}) = \mathcal{N}(\boldsymbol{x};\mu(\boldsymbol{h}),\sigma(\boldsymbol{h}))	
\end{align}
$$

Binary X values correspond to a Bernoulli distribution whose paramenters are given by a sigmoid output, here we use cross entropy loss to calculate the error. 

Discrete x values correspond to a softmax distribution, here we also use the cross entropy loss.

## Denoising Autoencoders (DAEs)

The denoising autoencoder (DAE) is an autoencoder that receives a corrupted data point as input and is trained to predict the original, uncorrupted data point as its output.

How does it work?

1. We define a corruption process to corrupt X: $C(\tilde{\boldsymbol{x} } | \boldsymbol{x})$, then we store both the corrupted and the non-corrupted data somewhere. 
2. We sample a training example $\boldsymbol{x}$ from the training data, and $\tilde{\boldsymbol{x} }$ from the corrupted version $C(\tilde{\textbf{x} }|\textbf{x} = \boldsymbol{x})$.
3. We then use $(\boldsymbol{x}, \tilde{\boldsymbol{x} })$ as a training example for estimating the autoencoder reconstruction distribution: $p_{\text{reconstruct} }(\boldsymbol{x}|\tilde{\boldsymbol{x} }) = p_{\text{decoder} }(\boldsymbol{x}|\boldsymbol{h})$, where the $\boldsymbol{h}$ is the output of the encoder $f(\tilde{\boldsymbol{x} })$.

Optimization: 

We can perform gradient-based approximate minimization such as minibatch gradient descent on the negative log likelihood: $-\log p_{\text{decoder} }(\boldsymbol{x}|\boldsymbol{h})$. So long as the encoder is deterministic, the denoising autoencoder is a feedforward network and may be trained with exactly the same techniques as any other feedforward network.

We can therefore view the DAE as performing stochastic gradient descent on the following expectation:

$$
-\mathbb{E}_{\textbf{x}\sim \hat{p}_\text{data} }\mathbb{E}_{\tilde{\textbf{x} }\sim C(\tilde{\textbf{x} }|\textbf{x})}\log p_{\text{decoder} }(\boldsymbol{x}|\boldsymbol{h} = f\left(\tilde{\boldsymbol{x} })\right)
$$

Where $\hat{p}_{\text{data} }(\textbf{x})$ is the training distribution

- Calculation example (to highlight notation mechanism)

	data points: $x_1 = 5, x_2 = 10$

	Corrupted versions: 

	x1: $\tilde{x}_{1,1} = 4, \tilde{x}_{1,2}=6$

	x2: $\tilde{x}_{2,1} = 9, \tilde{x}_{2,2}=11$

	Encoder-decoder gives:

	$f(4)=h_{1,1}=0.8, f(6)=h_{1,2}=1.2$

	$f(9)=h_{2,1}=1.8, f(11)=h_{2,2}=2.2$

	Assume log likelihood constructions are:

	$\log p_{\text{decoder} }(5|h_{1,1}) = -0.1, \log p_{\text{decoder} }(5|h_{1,2}) = -0.2$

	$\log p_{\text{decoder} }(10|h_{2,1}) = -0.3, \log p_{\text{decoder} }(10|h_{2,2}) = -0.4$

	Calculate objective:

	$$
	\begin{align*}
	& -\mathbb{E}_{\textbf{x}\sim \hat{p}_\text{data} }\mathbb{E}_{\tilde{\textbf{x} }\sim C(\tilde{\textbf{x} }|\textbf{x})}\log p_{\text{decoder} }(\boldsymbol{x}|\boldsymbol{h} = f\left(\tilde{\boldsymbol{x} })\right) =  \\ & -\frac{1}{2}(\frac{-0.1 + -0.2}{2} + \frac{-0.3 + -0.4}{2}) = \\ & - \frac{1}{2}(-0.15 + -0.35) = -\frac{1}{2} \cdot -0.5 = 0.25
	\end{align*}
	$$

So really, the inner expectation calculates the expectation term for each sample x in **x**. Empirically, this is the average.

Since |**x**| = 2, we do this twice. We then add these together and divide by 2 to get the outer mean (the empirical approximation to the outer expectation).

<img src="assets/Screenshot_2023-09-14_at_19.16.17.png" alt="Screenshot 2023-09-14 at 19.16.17.png" style="zoom: 33%;" />

### Score matching

Score matching is an alternantive to maximun likelihood.

- consistent estimations of probability distributions based on encouraging the model to have the same `score` as the data distribution at every training point $\boldsymbol{x}$.

In the context of denoising auto envoders, the score is a paricular gradient field:

$$
\begin{align}
\nabla_{\boldsymbol{x} } \log p (\boldsymbol{x})
\end{align}
$$

The gradient operator, $\nabla_{\boldsymbol{x} }$, is a vector containing the partial derivatives of the different vector components of **x** when applied to a function**.** These indicate different slopes along the different “directions”. 

Log(p(x)) is a function that maps $\boldsymbol{x}$ from $\mathbb{R}^{n} \rightarrow \mathbb{R}$. It is the logarithm of the probability density function (or mass if x is discrete).

Thus, when taking the gradient of log(p(x)), the result is a vector known as the “score” where each component is given by:

$$
\frac{\partial \log p(\boldsymbol{x})}{\partial x_i}
$$

The vector points in the direction in which log p(x) increases most rapidly.

- Why do we take the log of p(x)?
	1. **Numerical stability**: Probabilities can be very small numbers, and when you multiply small numbers, you risk numerical underflow. Taking the logarithm turns these multiplications into additions, which are numerically more stable.
	2. **Simplification of calculations**: Multiplying probabilities corresponds to adding their logarithms. This can simplify the math, especially when you're dealing with products of probabilities (as is often the case in likelihood functions).
	3. **Transforming products into sums**: In many statistical models, the likelihood involves a product of terms, one for each data point. Taking the log transforms this into a sum, which is often easier to work with. This is particularly useful for optimization algorithms like gradient descent, which prefer smooth landscapes.
	4. **Easier derivative calculation**: In many cases, taking the derivative of the log-likelihood is analytically easier than taking the derivative of the raw likelihood function. This is crucial for optimization.
	5. **Interpretability**: In some contexts, the log-likelihood has a more straightforward interpretation. For example, in logistic regression, the log-odds ratio is often more interpretable than the raw odds ratio.
	6. **Connection to information theory**: The logarithm of probabilities is related to information content (measured in bits if you use base-2 logarithms, or nats if you use natural logarithms). This has various theoretical advantages in understanding the behavior of algorithms.
	7. **Converting multiplicative factors to additive**: Some models have multiplicative factors that become additive after taking the logarithm, making it easier to separate different components of the model.

[Transformations](../math/transformations/transformations.md)

Learning the gradient field of $\log p_{\text{data} }$ is one way to learn the structure of $p_{\text{data} }$ itself. 

A very important property of DAEs is that their training criterion (with conditionally Gaussian p(x | h)) makes the autoencoder learn a vector field (g(f(x)) − x) that estimates the score of the data distribution. (figure 14.4)

## Learning manifolds with Autoencoders

[Manifolds](../math/topology_and_manifolds/differentiable_manifolds.md):

[https://mathworld.wolfram.com/Manifold.html](https://mathworld.wolfram.com/Manifold.html)

Like many other machine learning algorithms, autoencoders exploit the idea that data concentrates around a low-dimensional manifold or a small set of such manifolds. Some machine learning algorithms exploit this idea only insofar as that they learn a function that behaves correctly on the manifold but may have unusual behavior if given an input that is off the manifold. Autoencoders take this idea further and aim to learn the structure of the manifold.

### Manifolds:

An important characterization of a manifold is the set of its tangent planes. At a point x on a d-dimensional manifold, the tangent plane is given by d basis vectors that span the local directions of variation allowed on the manifold. As illustrated in **figure** **14.6**, these local directions specify how one can change x infinitesimally while staying on the manifold.

<img src="assets/Screenshot_2023-09-14_at_23.52.40.png" alt="Screenshot 2023-09-14 at 23.52.40.png" style="zoom: 33%;" />

<img src="assets/Screenshot_2023-09-14_at_23.58.43.png" alt="Screenshot 2023-09-14 at 23.58.43.png" style="zoom:33%;" />

All autoencoder training procedures involve a compromise between two forces:

1. Learning a representation h of a training example x such that x can be approximately recovered from h through a decoder. The fact that x is drawn from the training data is crucial, because it means the autoencoder need not successfully reconstruct inputs that are not probable under the data generating distribution.
2. Satisfying the constraint or regularization penalty. This can be an architectural constraint that limits the capacity of the autoencoder, or it can be a regularization term added to the reconstruction cost. These techniques generally prefer solutions that are less sensitive to the input.

Clearly, neither force alone would be useful. Copying the input to the output is not useful on its own, nor is ignoring the input. Instead, the two forces together are useful because they force the hidden representation to capture information about the structure of the data generating distribution. The important principle is that the autoencoder can afford to represent only the variations that are needed to reconstruct training examples. If the data generating distribution concentrates near a low-dimensional manifold, this yields representations that implicitly capture a local coordinate system for this manifold: only the variations tangent to the manifold around x need to correspond to changes in h = f(x).


💡 Hence the encoder learns a mapping from the input space x to a representation space, a mapping that is only sensitive to changes along the manifold directions, but that is insensitive to changes orthogonal to the manifold.

To understand why autoencoders are useful for manifold learning, it is instructive to compare them to other approaches. What is most commonly learned to characterize a manifold is a representation of the data points on (or near) the manifold. Such a representation for a particular example is also called its embedding. It is typically given by a low-dimensional vector, with less dimensions than the “ambient” space of which the manifold is a low-dimensional subset.

## Contractive autoencoder

This autoencoder uses a regularizer on the code $\boldsymbol{h}$ to make the derivatives as small as possible

$$
\Omega(\boldsymbol{h}) = \lambda \left\|\frac{\partial f(\boldsymbol{x})}{\partial \boldsymbol{x} }\right\|^2_F
$$

The penalty is the squared frobenius norm of the jacobian matrix of partial derivatives associated with the encoder function.

The name contractive arises from the way that the CAE warps space. Specifically, because the CAE is trained to resist perturbations of its input, it is encouraged to map a neighborhood of input points to a smaller neighborhood of output points. We can think of this as contracting the input neighborhood to a smaller output neighborhood. TO CLARIFY: the CAE is contractive only locally, all perturbations of a training point $\boldsymbol{x}$ are mapped near to $f(\boldsymbol{x})$. Globally, however, two different points $\boldsymbol{x}$ and $\boldsymbol{x}^{\prime}$ may be mapped to $f(\boldsymbol{x})$ and $f(\boldsymbol{x}^{\prime})$ wich are farther apart than their original points!

This makes sense because the jacobian basically contains all first-order partial derivatives of a vector-valued function. In the context of the encoder function: $f: \mathbb{R}^n \Rightarrow \mathbb{R}^m$, the jacobian matrix $\boldsymbol{J}_{ij}$ is the partial derivative of the i-th output with respect to the j-th input.

$$
\begin{align} 
\boldsymbol{J} = \left(\begin{array}{cccc}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{array}\right)
\end{align}
$$

This matrix provides a linear approximation that describes how small changes in the input $\boldsymbol{x}$ will affect the output $f(\boldsymbol{x}) = \boldsymbol{h}$. In the context of a Contractive Autoencoder, you want this matrix to have small values, which means that the function is not very sensitive to changes in its input—hence the "contractive" property.

As described, regularized autoencoders learn manifolds by balancing two opposing forces. In the case of the CAE, these two forces are reconstruction error and the contractive penalty Ω(h). Reconstruction error alone would encourage the CAE to learn an identity function. The contractive penalty alone would encourage the CAE to learn features that are constant with respect to x. The compromise between these two forces yields an autoencoder whose derivatives are mostly tiny. Only a small number of hidden units, corresponding to a small number of directions in the input, may have significant derivatives.

# Probabilistic approach to latent variable models

<img src="assets/Untitled.png" alt="Untitled" style="zoom:33%;" />

Joint distribution of Xs and latent variables Zs can be decomposed in terms of a simple distribution of our latent manifold p(z), which we simply take to be a normal distribution.

$$
p_\theta(x, z)=\underbrace{p_\theta(x \mid z)}_{\text {FFNN } } \underbrace{p(z)}_{\mathcal{N}(z \mid 0, I)}
$$

We have an inference step, that is finding the posterior distribution of $z$ given an $x$, which corresponds to reversing the conditionals using Bayes theorem:

$$
z \sim p(z|x)=\frac{p(x|z)p(z)}{p(x)}
$$

We have a likelihood step:

$$
p(x) = \int p(x|z)p(z)dz
$$

Learning: 

$$
\theta_{\text{ML} } = \operatorname{arg max}_\theta \sum_{i=1}^n\log p_\theta(x_i)
$$

All are very difficult computations!

Big breakthrough: we can use variational inference and deep learning models to define an encoder and a decoder. The variational approach will naturally lead to a formulation of an encoder.

This leads to the Variational autoencoder;

# Variational Autoencoders

Decoder

$$
p_\theta(x|z)=\mathcal{N}\left(x|\underbrace{\mu_\theta(z)}_{\text {FFNN } }, \underbrace{\operatorname{diag}\left(\sigma_\theta^2(z)\right)}_{\text {FFNN} }\right)
$$

Assume $x$ is continuous, then we can model $x$’s distribution with a gaussian normal distribution, and we use neural networks to describe the mean and the variance.

The specific value of the mean and variance will depend on the specific value of the latent variable

So our NN is now used as a mapping from $z$ to a mean value and a variance value.

Encoder:

$$
q_\phi(z|x)=\mathcal{N}\left(z|\underbrace{\mu_\phi(x)}_{\text{FFNN} }, \underbrace{\operatorname{diag}\left(\sigma^2_\phi(x)\right)}_{\text{FFNN} }\right)
$$

The encoder goes the other way around, it maps input $x$ to a mean and variance of $z$.

The encoder and decoder are thus two different NNs, there are NO parameter sharing between them.

**Variational objective:** 

Optimize

$$
\sum_i \mathcal{L}_{\theta,\phi}(x_i), \text{ w.r.t } \phi, \theta:
$$

$$
\begin{align*}\log p_\theta(x) & \geq \mathcal{L}_{\theta, \phi}(x) = \color{red} \int q_{\theta}(z|x)\log \left(\frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right)dz \\ & = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\mathbb{E}_q \log\text{likelihood} }+ \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p(z)}{q_\phi (z|x)}\right]}_{\text{regularization} }\end{align*}
$$

We can write a lower bound of the likelihood. It is still an integral over $z$, but it turns out, it is much easier to evaluate than the original integral.

This variational bound is written for one single example (coloured red). It involves an average of the encoder distribution of the log between the joint distribution of x and z from the generative model (decoder?) divided by the encoder function. 

We can decompose the log into two terms, the first term here is the expected value over the encoder distribution of the log-likelihood (this is a data fit term). it essentially measures how well we on average can reproduce the observations. So if we are good at reproducing, this term will be large.

And then we have a regularization term, it has the log of the prior distribution over the latent space divided by the encoder variational posterior distribution. This is like a KL divergence between $q$ and the prior. We know that that is minimum when the posterior is equal to the prior. This term would actually try to squeeze this q-distribution towards the prior distribution, and if it does that, then it has not learned anything about the data. 

<aside>
💡 So optimising the variational bound is a tradeoff between fitting the data and getting closer to the prior manifold!
The regularization term is essentially the KL divergence between the encoders distribution $q_\phi(z|x)$ and the prior $p(z)$.

- The original $\log p_\theta(x)$ is hard to compute but $\mathcal{L}_{\theta, \phi}(x)$ is a **tractable lower bound**

1. **Two Neural Networks**: One for encoding $x$ to $z$ and another for decoding $z$ to $x$, with no parameter sharing.
2. **Gaussian Assumptions**: Both the encoder and decoder output Gaussian distributions, parameterized by neural networks.
3. **Objective Function**: It's a balance between reconstruction quality and regularization.
4. **Tractability**: The variational lower bound makes an otherwise intractable problem more manageable.

## Handle integration by sampling

Remember:

$$
\log p_\theta(x) \geq \mathcal{L}_{\theta, \phi}(x) = \int q_{\theta}(z|x)\log \left(\frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right)dz 
$$

We can approximate:

$$
\mathcal{L}_{\theta, \phi}(x) \approx \frac{1}{R}\sum_{r=1}^R\log \frac{p_\theta(x|z^r)p(z^r)}{q_\phi(z^r|x)}, \\z^r = \mu_\phi(x)+\sigma_\phi(x) \otimes \epsilon^r
$$

Instead of using $z$ as a stochastic variable, we can use $\epsilon$ which is a normally distributed variable with $0$ mean and unit variance. It has no parameters and we have replaced all occurrences in the integral by this deterministic mean functions times the standard deviation times this noise we draw from this simple normal distribution

Instead of having the integral, we replace it by a sample, usually we set $R$ equal to 1 to get the optimal tradeoff between speed and accuracy.

Now we have a bound for one example, and then the log likelihood bound is the sum of all the examples and we can now take derivatives with respect to this bound with respect to theta and phi in order to at the same time optimise the likelihood, and also make the bound as tight as possible with regards to optimising in respect to the variational distribution. 

The variational distribution is of course limited in a way, that we have taken a quite simple normal distribution. If we have a lot of data it will be asymptotically normally, but we have ignored covariances between the latent variables due to the diagonal covariance matrix above.

**Encoder**  $z = \mu_\phi(x) + \sigma(x) \otimes \epsilon$

**Decoder**  $p_\theta(x|z) = \mathcal{N}\left(x|\mu_\theta(z),\operatorname{diag}(\sigma^2_\theta(z))\right)$

- We allow us to compute the conditional density of our observed observation $x$.

**Why do we call this an autoencoder?**

- It has both an encoder and a decoder.
- We map “x —> z —> x”

## Deep learning book:

How it works:

To generate a sample from the model, VAE draws a sample $\boldsymbol{z}$ from the coding distribution $p_{\text{model} }(\boldsymbol{z})$. This sample then run through a differentiable generator network $g(\boldsymbol{z})$. Finally $\boldsymbol{x}$ is sampled from a distribution $p_{\text{model} }(\boldsymbol{x};g(\boldsymbol{z})) = p_{\text{model} }(\boldsymbol{x}|\boldsymbol{z})$. 

However, during training, the approximate inference network (or encoder) $q(\boldsymbol{z}|\boldsymbol{x})$ is used to obtain $\boldsymbol{z}$ and $p_{\text{model} }(\boldsymbol{x}|\boldsymbol{z})$ is then viewed as the decoder network.

The key insight behind VAEs is that they can be trained by maximizing the variational lower bound $\mathcal{L}(q)$ associated with data point $\boldsymbol{x}$:

 

\\[
\begin{align} \mathcal{L}(q) & = \mathbb{E}_{\boldsymbol{z} \sim q(\boldsymbol{z} | \boldsymbol{x})\log p_{\text{model} } (\boldsymbol{z}|\boldsymbol{x}) + \mathcal{H}(q(\textbf{z}|\boldsymbol{x}))} \\ &= \mathbb{E}_{\boldsymbol{z} \sim q(\boldsymbol{z} | \boldsymbol{x})\log p_{\text{model} } (\boldsymbol{x}|\boldsymbol{z})} - D_{KL}(q(\textbf{z}|\boldsymbol{x})\|p_{\text{model} }(\textbf{z})) \\ & \leq \log p_{\text{model} }(\boldsymbol{x}). \end{align}
\\]

VAE Pytorch implementation:

[link](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing)
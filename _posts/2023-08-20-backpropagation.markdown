---
layout: post
comments: true
title:  "Deep Dive into Backpropagation in Neural Networks"
excerpt: "An in-depth exploration of backpropagation, the backbone of learning in neural networks."
date:   2023-08-20 22:00:00
category: "DL"
mathjax: true
---
# Backpropagation

## The four fundamental formulas

### Equation of error in the output layer L

# Fundamental equations and their proofs

## The four fundamental formulas

### Equation of error in the output layer L

$$
\begin{align}
\delta_j^L = \frac{\partial C}{\partial a_j^L}\sigma'(z_j^L)
\end{align}
$$

This expression tells, that the error of node j in layer L (the last layer) is a function of the partial derivative of the cost function with respect to the activation of node j in the last layer, as well as the how fast the activation function *σ* is changing with respect to \\(z_j^L\\).

where \\(z_j^L\\) is the weighted sum of incoming influences of nodes in the previous layer. In other words, \\(z_j^L\\) is node \\(j\\) in layer \\(L\\) before we have called the activation function \\(σ\\).

$$
\begin{align}z_j^L = \sum_k w_{jk}^La_k^{L-1}+b_j^L\end{align}
$$

Where: 

- \\(a_k^{L-1} = \sigma(z_k^{L-1})\\),
- \\(b_j^L\\) is a bias term for node \\(j\\) in layer \\(L\\),
- \\(w_{jk}^L\\) is a weight matrix linking influences from last layers nodes \\(k\\) to this layers nodes \\(j\\)

We can write the equation in matrix form:

$$
\delta^L = \nabla_aC\odot\sigma'(z^L)
$$

where: \\(\nabla_aC\\) is a vector containing the partial derivatives: \\(\frac{\partial C}{\partial a_j^L}\\)

### Equation of error in terms of the error in the next layer

$$
\begin{align}\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)\end{align}
$$

Transposing the weight matrix has the function of moving the error backward through the network. This gives us a measure of the error at the output of the \\(l\\)th layer in the network.

We then take the element wise product of the change in the activation function of layer \\(l\\), when we change \\(z^l\\).

This moves the error backward through the activation function in layer \\(l\\), giving us the error \\(δ^l\\) in the weighted input to layer \\(l\\).

**Takeaway:** by combining **(1)** and **(3)**, we can compute the error \\(δ^l\\) for any layer. We start with **(1)** to get the error of the last layer, then we use **(3)** to get the errors of the previous layers.

### Equation for rate of change of the cost w.r.t. any bias in the network

$$
\begin{align}
\frac{\partial C}{\partial b_j^l} = \delta_j^l
\end{align}
$$

This implies that the error \\(δ_j^l\\) is equal to the rate of change \\(\frac{\partial C}{\partial b_j^l}\\).

Since **(1)** and **(3)** have shown us how to compute \\(δ^l\\), we use the shorthand term:

$$
\begin{align*}
\frac{\partial C}{\partial b} = \delta
\end{align*}
$$

where it is understood that \\(δ\\) is being evaluated at the same neuron as the bias \\(b\\)

### Equation for rate of change of cost w.r.t. any weight in the network

$$
\begin{align}
\frac{\partial C}{\partial w_{jk}^l} = a_k^{l-1}\delta_j^l
\end{align}
$$

Tells us how to compute the partial derivatives of the weights linking node \\(k\\) to node \\(j\\). We observe that we have already learned how to compute each of the terms in the expression: \\(a_k^l − 1\\) and \\(δ_j^l\\).

We can write the expression in short:

$$
\begin{align*}
\frac{\partial C}{\partial w}=&a_\text{in} \delta_\text{out} \\ \\
\text{where} \\  & a_\text{in}: \text{is the activation of the neuron input to the weight} \\(w \\) \\
& \delta_\text{out}: \text{is the error of the neuron output from the weight.} \\( w \\)
\end{align*}
$$

**Interpretation if one is to use a sigmoid activation function:**

Meaning: when \\(a_{\text{in} }\\) is approximating \\(0\\), the gradient term \\(\frac{\partial C}{\partial w}\\) will also be small. Hence the network learns slowly from this neuron (its not changing much during gradient descent). In other words, one consequence of (BP4) is that weights output from low-activation neurons learn slowly.

> Summing up, we’ve learnt that a weight will learn slowly if either the input neuron is low-activation, or if the output neuron has saturated, i.e., is either high- or low-activation.

ReLu, is a way to eliminate this, since it won’t get saturated by reaching 1 or higher.

## Proofs of the four fundamental formulas

### EQ 1: Error in terms of last layer

$$
\begin{align*}
  \delta^L_j = \frac{\partial C}{\partial z^L_j}.
\end{align*}
$$

Applying `the chain rule` we can reexpress the partial derivative above in terms of partial derivatives with respect to the output activations:

$$
\begin{align*}
  \delta^L_j = \sum_k \frac{\partial C}{\partial a^L_k} \frac{\partial a^L_k}{\partial z^L_j},
\end{align*}
$$

where the sum is over all neurons \\(k\\) in the output layer. 

Of course, the output activation \\(a_k^L\\) of the \\(k\\)th neuron depends only on the weighted input \\(z_j^L\\) for the \\(j\\)th neuron when \\(k = j\\). And so \\(\frac{\partial a^L_k}{\partial z^L_j}\\) vanishes when \\(k ≠ j\\) As a result we can simplify the previous equation to:

$$
\begin{align*}
\delta^L_j = \frac{\partial C}{\partial a^L_j}\frac{\partial a^L_j}{\partial z^L_j}.
\end{align*}
$$

Since \\(a_j^L = σ(z_j^L)\\), we can write the partial derivative to the right as: \\(σ^{\prime}(z_j^L)\\).

Hence we get:

$$
\begin{align*}
\delta^L_j = \frac{\partial C}{\partial a^L_j} \sigma'(z^L_j) & & & & 
 \blacksquare
\end{align*}
$$

### EQ 2: Error in terms of layers prior to last layer

We rewrite \\(δ_j^l = \partial C/\partial z_j^l\\) in terms of \\(\delta_k^{l + 1} = \partial C/\partial z_k^{l + 1}\\).

To do this, we use the chain rule:

$$
\begin{align*}
  \delta^l_j & =  \frac{\partial C}{\partial z^l_j} \\
  & =  \sum_k \frac{\partial C}{\partial z^{l+1}_k} \frac{\partial z^{l+1}_k}{\partial z^l_j} \\ 
  & =  \sum_k \frac{\partial z^{l+1}_k}{\partial z^l_j} \delta^{l+1}_k,
\end{align*}
$$

To evaluate the first term on the last line, note that

$$
\begin{align*}
z^{l+1}_k = \sum_j w^{l+1}_{kj} a^l_j +b^{l+1}_k = \sum_j w^{l+1}_{kj} \sigma(z^l_j) +b^{l+1}_k.
\end{align*}
$$

Differentiating we obtain:

$$
\begin{align*}
\frac{\partial z^{l+1}_k}{\partial z^l_j} = w^{l+1}_{kj} \sigma'(z^l_j).
\end{align*}
$$

Substituting, we get:

$$
\begin{align*}
\delta^l_j = \sum_k w^{l+1}_{kj}  \delta^{l+1}_k \sigma'(z^l_j) & & & & \blacksquare
\end{align*}
$$

### EQ 3: Error w.r.t. bias

We remember **(3)**:

$$
\begin{align*}
\frac{\partial C}{\partial b_j^l} = \delta_j^l
\end{align*}
$$

we know that \\(\delta^j_l = \frac{\partial C}{\partial z^l_j}\\)

Using the chain rule:

$$
\begin{align*}
  \delta^l_j = \sum_k \frac{\partial C}{\partial b^l_k} \frac{\partial b^l_k}{\partial z^l_j},
\end{align*}
$$

Since \\(b_k^l\\) depends only on \\(z_k^l\\), all terms expect \\(j = k\\) is \\(0\\).

$$
\begin{align*}
  \delta^l_j = \frac{\partial C}{\partial b^l_j} \frac{\partial b^l_j}{\partial z^l_j},
\end{align*}
$$

We know that differentiating with respect to the bias, which has an additive relation to \\(z_j^l\\) $$z_j^l = ∑_k w_{jk}^la_k^l − 1 + b_j^l$$, the derivative \\(\frac{\partial b^l_j}{\partial z^l_j} = 1\\).

Therefore:

$$
\begin{align*}
  \delta^l_j = \frac{\partial C}{\partial b^l_j} & & & & \blacksquare
\end{align*}
$$

### EQ 4: Error w.r.t. weights

We remember **(4)**:

$$
\begin{align*}
\frac{\partial C}{\partial w_{jk}^l} = a_k^{l-1}\delta_j^l
\end{align*}
$$

We apply the chain rule:

$$
\begin{align*}
\frac{\partial C}{\partial w_{jk}^l} = \sum_k\frac{\partial C}{\partial z^l_k} \frac{\partial z^l_k}{\partial w_{jk}^l}
\end{align*}
$$

As previous, we know that for all other combinations but \\(j = k\\) the we get \\(0\\). Therefore, we can rewrite the expression:

$$
\begin{align*}
\frac{\partial C}{\partial w_{jk}^l} = \frac{\partial C}{\partial z^l_j} \frac{\partial z^l_j}{\partial w_{jk}^l}
\end{align*}
$$

since \\(\frac{\partial C}{\partial z^l_j}=\delta^l_j\\), and $$z_j^l = ∑_k w_{jk}^la_k^l − 1 + b_j^l$$ , the partial derivative is \\(\frac{\partial z^l_j}{\partial w_{jk}^l} = a_k^{l-1}\\).

Thus we can rewrite the expression above and finish the proof:

$$
\begin{align*}
\frac{\partial C}{\partial w_{jk}^l}= a_k^{l-1}\delta_j^l & & & & \blacksquare
\end{align*}
$$

$$
\begin{align}
z_j^L = \sum_k w_{jk}^La_k^{L-1}+b_j^L
\end{align}
$$

Where: 

- \\(a_k^{L-1} = \sigma(z_k^{L-1})\\),
- \\(b_j^L\\) is a bias term for node \\(j\\) in layer \\(L\\),
- \\(w_{jk}^L\\) is a weight matrix linking influences from last layers nodes \\(k\\) to this layers nodes \\(j\\)
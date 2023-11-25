---
layout: post
comments: true
title:  "Understanding Cross-Entropy Loss in Deep Learning"
excerpt: "Exploring the significance and mathematical formulation of cross-entropy loss, a cornerstone in neural network training."
date:   2023-08-20 22:00:00
category: "DL"
mathjax: true
---
# Cross entropy loss

# **Cross-entropy**

**How to address learning slowdown**

**Formula:**

$$
\begin{align}
C = -\frac{1}{n} \sum_x \left[y \ln a + (1-y ) \ln (1-a) \right]
\end{align}
$$

**Two properties:**

**The function is non-negative: C>0**

**Notice parameters in sum equation are negative, and the -1 outside the sum turns it positive. If the neuron's output is close to the desired output for all training inputs, x, then the cross-entropy will be close to zero**

**To see this, suppose for example that y=0 and a≈0 for some input x. This is a case when the neuron is doing a good job on that input. We see that the first term in the expression for the cost vanishes, since y=0, while the second term is just −ln(1−a)≈0.** 

**A similar analysis holds when y=1 and a≈1. And so the contribution to the cost will be low provided the actual output is close to the desired output.**

**The derivative of the cross-entropy function:**

$$
\begin{align}
  \frac{\partial C}{\partial w_j} & = & \frac{1}{n}
  \sum_x \frac{\sigma'(z) x_j}{\sigma(z) (1-\sigma(z))}
  (\sigma(z)-y).
\end{align}
$$

**With the sigmoid function as activation function it can be simplified to** 
\\(\frac{\partial C}{\partial w_j} =  \frac{1}{n} \sum_x x_j(\sigma(z)-y)\\)**.**

**Tells us that the learning rate of the weight is controlled by** \\(\sigma(z)-y\\)**, i.e. by the error in the output. The larger the error, the faster the learning.**

**Partial derivative for bias**

$$
\begin{align*} 
  \frac{\partial C}{\partial b} = \frac{1}{n} \sum_x (\sigma(z)-y).
\end{align*}
$$

**We can see the same tendency as the bullet points right above.**

**Cross-entropy generalized:**

$$
\begin{align}  C = -\frac{1}{n} \sum_x
  \sum_j \left[y_j \ln a^L_j + (1-y_j) \ln (1-a^L_j) \right].
\end{align}
$$

**where** \\(\boldsymbol{y}\\) **is a vector of desired outputs of each output node, and** \\(a^L_1,a^L_2,\dots\\) **are the actual output values**

**Notice that we sum over all of the output nodes as well in the generalized cross-entropy scenario**

> You can think of (63) as a summed set of per-neuron cross-entropies, with the activation of each neuron being interpreted as part of a two-element probability distribution* (Of course, in our networks there are no probabilistic elements, so they're not really probabilities.*) In this sense, (63) is a generalization of the cross-entropy for probability distributions.
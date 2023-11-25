---
layout: post
comments: true
title:  "Regularization Techniques in Deep Learning"
excerpt: "A comprehensive look at regularization methods, such as L2 regularization and dropout, and their role in reducing overfitting in neural networks."
date:   2023-08-20 22:00:00
category: "DL"
mathjax: true
---
# Regularization

# **Regularization (L2)**

**A way to reduce overfitting**

**There are multiple ways in which a neural net can overfit, and multiple antidotes**

**Increase the training sampleThis increases potential for learning and hence mitigates overfitting even for large NNReduce NN sizeThis will reduce overfittingReluctant method, because greater NN-size means more power (if not overfit)RegularizationCommon technique: `weight decay`**

**General expression of `weight decay`**

\\(C =C_0 + \frac{\lambda}{2n} \sum_w w^2\\)

**where C_0 is an arbitrary loss function.**

**Example**

**Weight decay onto cross-entropy**

$$
C = -\frac{1}{n} \sum_{xj} \left[ y_j \ln a^L_j+(1-y_j) \ln
(1-a^L_j)\right] + \frac{\lambda}{2n} \sum_w w^2.
$$

> Intuitively, the effect of regularization is to make it so the network prefers to learn small weights, all other things being equal. Large weights will only be allowed if they considerably improve the first part of the cost function. Put another way, regularization can be viewed as a way of compromising between finding small weights and minimizing the original cost function. The relative importance of the two elements of the compromise depends on the value of λ: when λ is small we prefer to minimize the original cost function, but when λ is large we prefer small weights.

**How to apply gradient decent in a regularized network?**

$$
\begin{align*}
\text{weights: }
  \frac{\partial C}{\partial w} & =  \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} w \\ \\ 
  \text{bias: } \frac{\partial C}{\partial b} & =  \frac{\partial C_0}{\partial b}.\end{align*}
$$

**Easy:**

> just use backpropagation, as usual, and then add \\(\frac{λ}{n}\\) w to the partial derivative of all the weight terms. The partial derivatives with respect to the biases are unchanged, and so the gradient descent learning rule for the biases doesn't change from the usual rule:
> \\(b \rightarrow  b -\eta \frac{\partial C_0}{\partial b}\\)

**Learning rule for weights:**

$$
\begin{align*}  w & \rightarrow  w-\eta \frac{\partial C_0}{\partial
    w}-\frac{\eta \lambda}{n} w \\ 
  & =  \left(1-\frac{\eta \lambda}{n}\right) w -\eta \frac{\partial
    C_0}{\partial w}.\end{align*} 
$$

**This is almost the same as we know it, except that we scale by a factor of** \\(1-\frac{\eta\lambda}{n}\\)**.**

**This rescaling is the `weight decay` since it makes the weights smaller.**

> At first glance it looks as though this means the weights are being driven unstoppably toward zero. But that's not right, since the other term may lead the weights to increase, if so doing causes a decrease in the unregularized cost function.

**How it works in a mini-batch:**

$$
w \rightarrow \left(1-\frac{\eta \lambda}{n}\right) w -\frac{\eta}{m}
  \sum_x \frac{\partial C_x}{\partial w}
$$

**Exactly the same as previously, just with the same weight decay factor as before.**

**Biases remain the same:** \\(b \rightarrow b - \frac{\eta}{m} \sum_x \frac{\partial C_x}{\partial b}\\)**, where the sum is over training examples x in the mini-batch.**

## Why does regularization reduce overfitting?

**Empirically, we can observe it. Common story: smaller weights can be understood as lower complexity, providing more simple and powerful explanation for the data.**

**Now to the real answer:**

**Let's see what this point of view means for neural networks. Suppose our network mostly has small weights, as will tend to happen in a regularized network. The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data. Think of it as a way of making it so single pieces of evidence don't matter too much to the output of the network. Instead, a regularized network learns to respond to types of evidence which are seen often across the training set. By contrast, a network with large weights may change its behaviour quite a bit in response to small changes in the input. And so an unregularized network can use large weights to learn a complex model that carries a lot of information about the noise in the training data. In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data. The hope is that this will force our networks to do real learning about the phenomenon at hand, and to generalize better from what they learn.**

***Simple is not always correct:***

**With that said, this idea of preferring simpler explanation should make you nervous. People sometimes refer to this idea as "Occam's Razor", and will zealously apply it as though it has the status of some general scientific principle. But, of course, it's not a general scientific principle. There is no *a priori* logical reason to prefer simple explanations over more complex explanations. Indeed, sometimes the more complex explanation turns out to be correct.**

**Thus there is no convincing theory as to why regularization is better. It oftentimes just is.**

## **Other forms of regularization / reduction mechanisms for overfitting**

**L1 regularization - similar to L2, but is less punishing for higher weights, and more punishing for lower weights**

**`Dropout` a mechanism that artificially leaves out half of the neurons in the forward pass, iteratively. In a way, it can be viewed as multiple NN combined in the end, and hence all weights are divided by 2, since their power should be reduced, when all nodes in the net are used.**

**This mechanism has had huge success in increasing predictability [link to article](https://arxiv.org/pdf/1207.0580.pdf)**

**`artificially expand training data`a mechanism to increase learning and reduce overfitting by manipulating existing data.**

**For the MNIST data set, this could for example be done by rotating some letters by 15 degreesThis will result in a similar but different pixel image, and thus is valid for training. Has had huge success in increasing predictions [link to article](https://ieeexplore.ieee.org/document/1227801) Also introduced "elastic distortions" - minor distortions in the images to reach a prediction accuracy of 99.3%**
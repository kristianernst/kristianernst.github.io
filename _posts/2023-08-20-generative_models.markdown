---
layout: post
comments: true
title:  "Generative models"
excerpt: "Generative models"
date:   2023-08-20 22:00:00
category: "Architectures"
mathjax: true
---


### What are Generative Models?

Generative models are a class of statistical models that aim to learn the underlying distribution of the data. In other words, they try to model P(X), the probability distribution from which the data X is generated. Once trained, a generative model can generate new data samples that are similar to the training data.

### Types of Generative Models

1. **Gaussian Mixture Models (GMMs)**: These are a simple form of generative model that assume data is generated from a mixture of several Gaussian distributions.
2. **Hidden Markov Models (HMMs)**: These are used for sequential data and assume that the data is generated from a sequence of hidden states.
3. **Generative Adversarial Networks (GANs)**: These consist of two neural networks, a generator and a discriminator, that are trained together. The generator tries to generate data, while the discriminator tries to distinguish between real and generated data.
4. **Variational Autoencoders (VAEs)**: These are a type of autoencoder that is designed to not just reconstruct the input data but also to model its underlying probability distribution.
5. **Bayesian Networks**: These are graphical models that represent the conditional dependencies among a set of variables.

### How Do They Work?

Generative models often work by introducing latent variables ***h***, which are not observed but capture the underlying structure in the data. The model then defines a joint probability P(x,h) over the observed ***x*** and latent ***h*** variables.

### Why Are They Useful?

1. **Data Generation**: They can generate new data that is similar to the training data.
2. **Feature Learning**: They can learn useful features of the data automatically, which can then be used for other tasks.
3. **Anomaly Detection**: By learning the distribution of 'normal' data, they can identify outliers or anomalies.
4. **Understanding Data**: They can reveal the underlying structure or patterns in the data.


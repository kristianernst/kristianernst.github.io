---
layout: post
comments: true
title:  "Baeysian statistics"
excerpt: "Bayesian statistics consider all possible values of θ when making a prediction."
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

Bayesian statistics consider all possible values of \\(\boldsymbol{\theta}\\) when making a prediction.

> The Bayesian uses probability to reflect degrees of certainty of states of knowledge. The dataset is directly observed and so is not random. On the other hand, the true parameter **θ** is unknown or uncertain and thus is represented as a random variable.

> Before observing the data, we represent our knowledge of **θ** using the prior probability distribution, p(**θ**) (sometimes referred to as simply “the prior”).

> Generally, the machine learning practitioner selects a prior distribution that is quite broad (i.e. with high entropy) to reflect a high degree of uncertainty in the value of **θ** before observing any data. For example, one might assume a priori that **θ** lies in some finite range or volume, with a uniform distribution. Many priors instead reflect a preference for “simpler” solutions (such as smaller magnitude coefficients, or a function that is closer to being constant).

We leverage bayes rule in bayesian statistics: [**Bayes Rule**](https://ernst-hub.github.io/math/2023/08/20/bayes_rule/)

If we have a set of data samples: {x_1, … , x_m}, we can recover the effect of data on our belief about \\(\boldsymbol{\theta}\\) by combining the data likelihood \\(p(x^{(1)}, \dots,  x^{(m)} \mid \boldsymbol{\theta})\\) with the prior via bayes rule:

$$
\begin{align}
p\left(\boldsymbol{\theta} \mid x^{(1)}, \dots, x^{(m)}\right) = \frac{p\left(x^{(1)},\dots,x^{(m)} \mid \boldsymbol{\theta}\right)p(\boldsymbol{\theta})}{p\left(x^{(1)},\dots,x^{(m)}\right)}
\end{align}
$$

In the scenarios where Bayesian estimation is typically used, the prior begins as a relatively uniform or Gaussian distribution with high entropy, and the observation of the data usually causes the posterior to lose entropy and concentrate around a few highly likely values of the parameters.

### Relating to MLE

Relative to maximum likelihood estimation, Bayesian estimation offers two important differences. 

First, unlike the maximum likelihood approach that makes predictions using a point estimate of **θ**, the Bayesian approach is to make predictions using a full distribution over **θ**.

For example, after observing \\(m\\) examples, the predicted distribution over the next data sample \\(x^{(m+1)}\\) is given by:

$$
p\left(x^{(m+1)} \mid x^{(1)},\dots,x^{(m)}\right) = \\  \int p\left(x^{(m+1)}\mid\boldsymbol{\theta} \right) p\left(\boldsymbol{\theta}\midx^{(1)},\dots,x^{(m)}\right)d\boldsymbol{\theta}
$$

Here each value of **θ** with positive probability density contributes to the prediction of the next example, with the contribution weighted by the posterior density itself. After having observed {x(1), . . . , x(m)}, if we are still quite uncertain about the value of **θ**, then this uncertainty is incorporated directly into any predictions we might make.


💡 The beauty about bayesian statistics is that we directly incorporate uncertainty into the predictions.



- **Calculation example: Predicting the next flip of a coin**

	To predict the next flip using bayesian statistics we deal with: 

	- a prior,
	- a likelihood,
	- and a posterior.

	prior

	We assume a Beta distribution for the prior \\(\theta\\)

	$$
	\text{Prior}: p(\theta) = \text{Beta}(\theta;\alpha=1, \beta=1)
	$$

	Likelihood

	*Likelihood for a single flip*

	the likelihood \\(p(x \mid \theta)\\) for a single coin flip \\(x\\) is:

	- \\(\theta\\) if heads
	- \\(1-\theta\\) if tails

**Likelihood for Multiple flips**

Assume each flip is independent of the others, we can calculate the joint likelihood by multiplying the likelihoods for each individual flip.

For our sequence: H, T, H, the likelihood would be:

$$
p(H,T,H\mid\theta) = p(H\mid\theta)\times p(T\mid\theta)\times p(H\mid\theta)
$$

substituting the likelihood for each flip:

$$
\theta \times (1 - \theta) \times \theta = \theta^2 (1-\theta)
$$

Posterior:

$$
\text{Posterior:} \ p(\theta \mid \text{data}) =  \frac{\theta^2(1-\theta)\times 1}{Z}=\frac{\theta^2(1-\theta)}{Z}
$$

we assume here that 

For a Beta distribution with α=1 and β=1, the distribution is uniform, meaning p(θ)=1 for *θ* in [0, 1]. That's why the calculations seemed to assume a uniform prior; it's a special case of the Beta distribution.

Also, Z is the scaling factor that ensures that the posterior integrates to 1.

$$
Z = p\left(x^{(1)}, \dots, x^{(m)}\right) = \int p\left(x^{(1)},\dots,x^{(m)},\theta\right)d\theta
$$

Which can be expanded to:

$$
Z = \int p\left(x^{(1)},\dots,x^{(m)}\mid\theta\right)p(\theta)d\theta
$$

For our example, we have that:

$$
\begin{align*}Z & = \int_0^1\theta^2(1-\theta)d\theta = \int_0^1 \theta^2d\theta - \int_0^1 \theta^3d\theta \\ & = \frac{\theta^3}{3} \times 1 - \frac{\theta^4}{4} \times 1  = \frac{1^3}{3}-\frac{1^4}{4} = \frac{1}{3} - \frac{1}{4} = \frac{4}{12} - \frac{3}{12} = \frac{1}{12} \end{align*}
$$

CALCULATION

When we want to estimate the likelihood of Head, we can now run the calculations:

$$
p\left( x^{(4)} = H\mid\text{data}\right) = \int_0^1\theta\times \frac{\theta^2(1-\theta)}{\frac{1}{12} }d\theta \\= \int_0^1 \theta \times 12 \times \theta^2(1-\theta)d\theta = 12\int_0^1\theta^3(1-\theta)d\theta
$$

We split up the integral using the rule: F(a-b) = F(a) - F(b)

$$
\begin{align}p\left( x^{(4)} = H\mid\text{data}\right) & =  12\int_0^1\theta^3 - \theta^4d\theta \\ & = 12\int_0^1\theta^3d\theta - 12\int_0^1\theta^4d\theta \\ &= (12\times \frac{\theta^4}{4}\times 0 + 12 \times \frac{\theta^4}{4} \times 1) - (12\times \frac{\theta^5}{5}\times 0 + 12 \times \frac{\theta^5}{5} \times 1) \\ 
& = 3 \theta^4 - \frac{12\theta^5} {5} \\ 
& = 3 \times 1^4 - \frac{12}{5}\times 1^5 = 3 - \frac{12}{5} \\  
& = \frac{15}{5} - \frac{12}{5} \\ 
& = 3/5 = 0.6 = 60\%
\end{align}
$$

The second important difference between the Bayesian approach to estimation and the maximum likelihood approach is due to the contribution of the Bayesian prior distribution The prior has an influence by shifting probability mass density towards regions of the parameter space that are preferred *a priori*.

### Maximum A **Posteriori** (MAP) Estimation

While we can get the whole probability distribution of \\(\theta\\), most operations involving the Bayesian posterior are intractable, i.e. they are computationally expensive! Therefore single-point estimates are preferred in many cases. 

However, we can benefit from bayesian statistics to let a prior influence the way we retrieve the single-point estimate. Rather than returning to MLE, we can utilize MAP. 

MAP chooses the point of maximal posterior probability (or maximal density in the continuous case):

$$
\begin{align}
\boldsymbol{\theta}_{MAP} = \operatorname{arg max}_{\boldsymbol{\theta} } \log p(\boldsymbol{x} \mid \boldsymbol{\theta})+ \log p (\boldsymbol{\theta})
\end{align}
$$

We recognize, above on the right hand side, log p(x \mid θ), i.e. the standard log- likelihood term, and log p(θ), corresponding to the prior distribution.
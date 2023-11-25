---
layout: post
comments: true
title:  "Denosing Diffusion Probabilistic Models"
excerpt: "SD one of the OG papers."
date:   2023-08-20 22:00:00
category: "SD"
mathjax: true
---

# Denoising diffusion probabilistic models

## Background

<img src="/assets/sd/image-20231005224728442.png" alt="image-20231005224728442" style="zoom:40%;" />

Diffusion models are latent variable models of the form \\(p_\theta\left(\mathbf{x}_0\right):=\int p_\theta\left(\mathbf{x}_{0: T}\right) d \mathbf{x}_{1: T}\\), where \\(\mathbf{x}_1, \ldots, \mathbf{x}_T\\) are latents of the same dimensionality as the data \\(\mathbf{x}_0 \sim q\left(\mathbf{x}_0\right)\\). The joint distribution \\(p_\theta\left(\mathbf{x}_{0: T}\right)\\) is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at \\(p\left(\mathbf{x}_T\right)=\mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)\\) :
$$
p_\theta\left(\mathbf{x}_{0: T}\right):=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right), \quad p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right):=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)
$$
What distinguishes diffusion models from other types of latent variable models is that the `approximate posterior` \\(q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)\\), called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule \\(\beta_1, \ldots, \beta_T\\) :
$$
q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right):=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right), \quad q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right):=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)
$$
Training is performed by optimizing the usual variational bound on negative log likelihood:
$$
\mathbb{E}\left[-\log p_\theta\left(\mathbf{x}_0\right)\right] \leq \mathbb{E}_q\left[-\log \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\right]=\mathbb{E}_q\left[-\log p\left(\mathbf{x}_T\right)-\sum_{t \geq 1} \log \frac{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}\right]=: L
$$
The forward process variances \\(\beta_t\\) can be learned by reparameterization or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in \\(p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\\), because both processes have the same functional form when \\(\beta_t\\) are small. A notable property of the forward process is that it admits sampling \\(\mathbf{x}_t\\) at an arbitrary timestep \\(t\\) in closed form: using the notation \\(\alpha_t:=1-\beta_t\\) and \\(\bar{\alpha}_t:=\prod_{s=1}^t \alpha_s\\), we have
$$
q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)
$$


Efficient training is therefore possible by optimizing random terms of \\(L\\) with stochastic gradient descent. Further improvements come from variance reduction by rewriting \\(L(3)\\) as:
$$
\mathbb{E}_q[\underbrace{D_{\mathrm{KL} }\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p\left(\mathbf{x}_T\right)\right)}_{L_T}+\sum_{t>1} \underbrace{D_{\mathrm{KL} }\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)}_{L_{t-1} } \underbrace{-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0}]
$$
Equation (5) uses KL divergence to directly compare \\(p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\\) against forward process posteriors, which are tractable when conditioned on \\(\mathbf{x}_0\\) :
$$
\begin{aligned}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) &=\mathcal{N}\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu} }_t\left(\mathbf{x}_t, \mathbf{x}_0\right), \tilde{\beta}_t \mathbf{I}\right) \\
\end{aligned}
$$
$$
\begin{aligned} \text { where } \quad \tilde{\boldsymbol{\mu} }_t\left(\mathbf{x}_t, \mathbf{x}_0\right) & :=\frac{\sqrt{\bar{\alpha}_{t-1} } \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0+\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t \quad \text { and } \quad \tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1} }{1-\bar{\alpha}_t} \beta_t \end{aligned}
$$

Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

### Notes to background

-  \\(p_\theta\\) is the probability distribution parameterised by \\(\theta\\) (the parms we are going to learn)

-  \\(\textbf{x}_0\\) is the observed data, it is this underlying data-distribution we want to model

-  \\(\textbf{x}_{1:T}\\) are the latent variables, they are hidden factors that help generate \\(\textbf{x}_0\\). \\(\textbf{x}_t\\) represents the state of the data at time t as it undergoes a diffusion process.

-  \\(\int \dots d\textbf{x}_{1:T}\\) is the integral over all possible values of \\(\textbf{x}_{1:T}\\). We are summing up contributions from all possible latent variables to get the final model.

-  \\(\textbf{x}_0 \sim q(\textbf{x}_0)\\) says that the observed data \\(\textbf{x}_0\\) comes from some distribution q. 

-  \\(p_\theta(\textbf{x}_{0:T})\\)  is the joint distribution.In our case, represents the joint distribution over the observed data and the entire sequence of latent states.

-  This needs some explanation:
     - If \\(|\textbf{x}_t|=5\\) then a joint distribution at each state will be a join of 5 different probability distributions. 
     	- So for all \\(t, \ t\in {1,\dots,T}\\) we join these 5 different probability distributions. hence we get \\(5 \times (T+1)\\) dimensional probability distribution (since we take \\(\textbf{x}_0\\) to \\(\textbf{x}_T\\)).
     	- One caveat, as defined by (1) the different states are not independent dimensions, they are tied together through learned transition distributions and the initial distribution \\(\textbf{x}_T\\). 
     	- you can think of it as a joint distribution in 5×(T+1) dimensions, but it's a complicated one where each subset of 5 dimensions is linked to the next via a learned, conditional Gaussian distribution.
     -  Joint distribution:<img src="/assets/sd/RCPxe.png" alt="How To Find Joint Probability Distribution In R" style="zoom: 70%;" />

-  Why  \\(p_\theta\left(\mathbf{x}_0\right):=\int p_\theta\left(\mathbf{x}_{0: T}\right) d \mathbf{x}_{1: T}\\) makes sense:

-  lets say we have two variables, \\(a\\) and \\(b\\)
  - \\(p(a)=\int p(a,b)db = \int p(a|b) \times p(b) db\\)
  - This is a fundamental definition in probability theory. The joint probability of \\(a\\) and \\(b\\) 1occurring can be found by first looking at the likelihood that \\(a\\) occurs given \\(b\\) and then multiplied by the likelihood that \\(b\\) occurs.
  - The integration sums up the influence of all possible values of b on a, thereby eliminating the condition, ultimately giving us the marginal distribution \\(p(a)\\)

-  The approximate posterior \\(q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)\\): is basically derived by multiplying joint probability distributions conditioned by previous distributions. 
     -  It is how we go from \\(\textbf{x}_0\\) to \\(\textbf{x}_T\\). We iteratively apply Gaussian noise. 
     -  This Gaussian noise is defined as : \\(q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right):=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)\\)
     -  It basically says that state \\(\textbf{x}_t\\) follows a gaussian distribution with a mean determined by a beta parameter and the previous vector, and a variance determined by the beta parameter.
     -  The diagonal of beta is to ensure that the gaussian distributions are independent.

-  The loss term: KL-divergence essentially tells us how far the approximate posterior \\(q\\) are from the true posterior \\(p_\theta\\) at different transitions \\(t\\). 

     -  Rao-Blackwell theorem is often used to improve the efficiency of an estimator by reducing its variance.  Basically, because both posteriors are Gaussian, their KL divergence can be calculated with a closed-form expression rather than Monte Carlo estimates, which is more computationally efficient.
     -  Formula:  \\(D_{KL}(p \| q) = \frac{1}{2} \left[ \text{Tr}(\Sigma_q^{-1} \Sigma_p) + (\mu_q - \mu_p)^T \Sigma_q^{-1} (\mu_q - \mu_p) - k + \log \left( \frac{\det \Sigma_q}{\det \Sigma_p} \right) \right]\\), where \\(\mu\\) are means and \\(\Sigma\\) are covariances.

-  Conditioning \\(q(\textbf{x}_{t-1}|\textbf{x}_t, \textbf{x}_0)\\) on both the current state t and the original data. 

     -  The model aims to learn how to reverse this diffusion process, starting from the noisy x_T to recover a denoised or "cleaned" version of x_0.
     -  \\(\textbf{x}_0\\) serves as a critical anchor point for the whole diffusion process. The conditioning on \\(\textbf{x}_0\\) can be thought of as encoding how each subsequent state should "remember" this initial clean state as it undergoes transformation. It serves to guide the reverse diffusion process.
     -  Of course, at this state, it is the predicted \\(\textbf{x}_0\\) and not the actual original data. This is explained at equation 15 in the paper.
     -  Conditioning on \\(\textbf{x}_t\\) is used to influence the estimate at \\(\textbf{x}_{t-1}\\). It's like using your most recent observation to inform your next move.

-  The effect of \\(\alpha\\) and \\(\beta\\) in (6): 

     -  Since \\(\beta\\) and \\(\alpha\\) are constrained by each other, a high \\(\alpha\\) results in a low \\(\beta\\)	

     -  High values of \\(\alpha\\) results in the model paying more attention to the current state relative to the original data
     -  Low values of \\(\beta\\) results in lower variance across the diagonal matrix.

💡 We can go from the original image \\(\textbf{x}_0\\) to any image \\(\textbf{x}_t\\) without “traversing the chain”, by using these parameters as demonstrated by equation (4).

💡 We know all parameters for (2), so really we just need a neural network to learn (1) after deciding on the settings for (2).

💡 While we theoretically can calculate \\(p_\theta\left(\mathbf{x}0\right):=\int p_\theta\left(\mathbf{x}_{0: T}\right) d \mathbf{x}_{1: T}\\), the problem becomes *intractable* for large data (which we have, so we need an approximate estimate). This estimate is achieved with (5)

## Training and sampling

### <img src="/assets/sd/image-20231007134048374.png" alt="image-20231007134048374" style="zoom:50%;" />

To represent the mean \\(\boldsymbol{\mu}_\theta(\textbf{x}_t,t)\\), we propose a specific parameterization motivated by the following analysis of \\(L_t\\). With \\(p_\theta(\textbf{x}_{t-1}|\textbf{x}_t) = \mathcal{N}(\textbf{x}_{t-1};\boldsymbol{\mu}_\theta(\textbf{x}_{t}, t), \sigma^2_t\boldsymbol{I})\\) we can write the loss:
$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma^2_t}\left\| \tilde{\boldsymbol{\mu} }_t(\textbf{x}_t, \textbf{x}_0) - \boldsymbol{\mu}_\theta(\textbf{x}_t, t)\right\|^2\right] + C
$$
\\(C\\) is a constant that does not depend on \\(\theta\\). 

The most straightforward parameterization of \\(\boldsymbol{\mu}_\theta\\) is a model that predicts \\(\tilde{\boldsymbol{\mu} }_t\\) (the forward process posterior mean). We can expand equation (8) by reparameterizing  (4) as: \\(\textbf{x}_t(\textbf{x}_0, \boldsymbol{\epsilon}) = \sqrt{\bar{\alpha}_t}\textbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}\\) for \\(\boldsymbol{e} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\\) and applying the forward process posterior formula (7):
$$
$$
\begin{align}
L_{t-1} - C &= \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon} } \left[\frac{1}{2\sigma^2_t} \left\| \tilde{\boldsymbol{\mu}_t} \left(\textbf{x}_t(\textbf{x}_0, \boldsymbol{\epsilon}), \frac{1}{\sqrt{\bar{\alpha}_t} }(\textbf{x}_t(\textbf{x}_0, \boldsymbol{\epsilon}) - \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}) - \boldsymbol{\mu}_\theta(\textbf{x}_t,(\textbf{x}_0,\boldsymbol{\epsilon}),t) \right) 		\right\|^2	  \right] \\
& = \mathbb{E}_{\textbf{x}_0, \boldsymbol{\epsilon} } \left[\frac{1}{2\sigma^2_t} \left\| \frac{1}{\sqrt{\alpha_t} } \left(\textbf{x}_t(\textbf{x}_0, \boldsymbol{\epsilon}) - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t} }\boldsymbol{\epsilon} \right) - \boldsymbol{\mu}_\theta(\textbf{x}_t,(\textbf{x}_0,\boldsymbol{\epsilon}),t)		\right\|^2	  \right]
\end{align}
$$
$$
Equation 10 reveals that \\(\boldsymbol{\mu}_\theta\\) must predict \\(\frac{1}{\sqrt{\alpha_t} } \left(\textbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t} }\boldsymbol{\epsilon} \right)\\) given \\(\textbf{x}_t\\). Since \\(\textbf{x}_t\\) is available as input to the model, we may choose the parameterization:
$$
\boldsymbol{\mu}_\theta(\textbf{x}_t, t) = \tilde{\boldsymbol{\mu} }\left(\textbf{x}_t, \frac{1}{\sqrt{\alpha_t} } \left(\textbf{x}_t - {\sqrt{1-\bar{\alpha}_t} }\boldsymbol{\epsilon}_\theta(\textbf{x}_t) \right)\right) = \frac{1}{\sqrt{\alpha_t} } \left(\textbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t} }\boldsymbol{\epsilon}(\textbf{x}_t,t) \right)
$$
Where \\(\epsilon_\theta\\) is a function approximator intended to predict \\(\boldsymbol\epsilon\\) from  \\(\textbf{x}_t\\). To sample \\(\textbf{x}_{t-1} \sim p_\theta(\textbf{x}_{t-1}|\textbf{x}_t)\\) is to compute: \\(\textbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t} } \left(\textbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t} }\boldsymbol{\epsilon}(\textbf{x}_t,t) \right) + \sigma_t \textbf{z}\\) where \\(\textbf{z}\sim\mathcal{N}(\boldsymbol 0,\boldsymbol I)\\).

The complete sampling procedure (Algorithm 2) resembles Langevin dynamics with \\(\boldsymbol{\epsilon}_\theta\\) a learned gradient of the data density. Furthermore, with the parameterization (11) eq(10) simplifies to:
$$
$$
\begin{align}
\mathbb{E}_{\textbf{x}_0, \boldsymbol \epsilon}\left[\frac{\beta^2_t}{2\sigma^2_t\alpha_t(1-\bar\alpha_t)} \left\| \boldsymbol\epsilon - \boldsymbol \epsilon_\theta \left(\sqrt{\bar\alpha}\textbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon, t \right) \right\|^2\right]
\end{align}
$$
$$
which resembles denoising score matching over multiple noise scales indexed by t. As eq 12 is equal to (one term of) the variational bound for the Langebvin-like reverse process (11), we see that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.

==To summarize, we can train the reverse process mean function approximator \\(\boldsymbol{\mu}_\theta\\) to predict \\(\tilde{\boldsymbol{\mu} }_t\\) , or by modifying its parameterization, we can train it to predict \\(\boldsymbol\epsilon\\)==. We have shown that the \\(\boldsymbol \epsilon\\)-prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model's variational bound to an objective that resembles denoising score matching.

Nonetheless, it is just another parameterization of \\(p_\theta(\textbf{x}_{t-1}|\textbf{x}_t)\\), so we verify its effectiveness in Section 4 in an ablation where we compare predicting \\(\boldsymbol \epsilon\\) against predicting \\(\tilde{\boldsymbol{\mu} }_t\\).
















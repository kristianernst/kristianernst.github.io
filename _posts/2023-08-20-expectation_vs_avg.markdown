---
layout: post
comments: true
title:  "Expectation vs average"
excerpt: "Expectation vs average"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---


Average: 

For a random variable X = {x_1, …, x_m}, the average \\(\bar{x}\\) is given:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^m x_i
$$

The expectation E[*X*] of a random variable X is a measure of the "average" value that X takes on when sampled from its distribution.

For a discrete random variable

\\(\mathbb{E}[X] = \sum_xx\cdot p(x)\\), where p(x) is the probability mass function

For a continuous random variable

\\(\mathbb{E}[X] = \int x\cdot f(x) dx\\), where f(x) is the probability density function

Difference between average and expectation:

1. **Finite vs. Infinite**: Averages are computed over finite datasets, while expectations are computed over all possible values of a random variable, which could be infinite.
2. **Empirical vs. Theoretical**: The average is an empirical measure computed from actual data. The expectation is a theoretical measure based on the underlying probability distribution.
3. **Sample vs. Population**: The average is often a "sample average" used to estimate the "population average," which is the expectation.

So, when you have a finite dataset X that you assume is drawn from some distribution p(x), the average serves as an empirical estimate for the theoretical expectation E[X].
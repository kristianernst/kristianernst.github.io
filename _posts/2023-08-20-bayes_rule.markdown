---
layout: post
comments: true
title:  "Bayes Rule"
excerpt: "Simple look"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Bayes Rule

$$
p(a|b) = \frac{p(b|a)\cdot p(a)}{p(b)}
$$

The conditional probability of a given b is equal to the conditional probability of b given a, times the probability of a, scaled by the probability of b. 

### proof:

$$
\begin{align}p(a|b) = \frac{p(a\cap b)}{p(b)}\end{align}
$$

Similarly:

$$
p(b|a) = \frac{p(b\cap a)}{p(a)}
$$

Rearranging the terms:

$$
p(a\cap b) = p(a|b)\cdot p(b)
$$

$$
p(a\cap b) = p(b|a) \cdot p(a)
$$

Therefore, we can re-express (1):

$$
p(a|b) = \frac{p(a\cap b)}{p(b)} = \frac{p(b|a)\cdot p(a)}{p(b)} \quad \blacksquare
$$

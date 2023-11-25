---
layout: post
comments: true
title:  "Kullback-Liebler Divergence"
excerpt: "The KL divergence is a measure of how one probability distribution diverges from a second expected probability distribution."
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

The KL divergence is a measure of how one probability distribution diverges from a second expected probability distribution.

The divergence of two probability distributions \\(p(x)\\) and \\(q(x)\\) can be expressed as:

$$
D_{KL} (p \| q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
$$

For continuous distributions, the sum is replaced by an integral
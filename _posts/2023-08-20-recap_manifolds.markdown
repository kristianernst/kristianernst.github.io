---
layout: post
comments: true
title:  "Recap 1: Constructing manifolds?"
excerpt: "An insightful exploration of Recap 1: Constructing manifolds?"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Recap 1: Constructing manifolds?

## Quick on topological manifolds

So a manifold is a topological space, which can be charted. These charts are part of an Atlas. 

\\((X, \mathcal{T}_X, \mathcal{A})\\). The charts map points or regions from the manifold to a euclidean space with the usual topology as explained in: 

[Topological manifolds](topological_manifolds.md)

- It is possible because a manifold is a topological space that is locally Euclidean

We remember the atlas was composed of charts \\((U, \gamma)\\), where U is an open set of the manifold and \\(\gamma\\) is

the mapping from the manifold to the euclidean space. The \\(\gamma\\)’s must be homeomorphic.

## Quick on differentiable manifolds
For a manifold to be differentiable, for any open set \\(U\\), we must be able to find a smooth transition function. \\(\gamma_i \circ \gamma_j^{-1} (\gamma_j(U_j))\\)

We know from the fact \\(\gamma\\)’s are homeomorphic that they are continuous, therefore, the composition of the two functions \\(\gamma\\) are continuous, \\(\gamma_i \circ \gamma_j^{-1}\\) is also continuous.

So let us just call \\(\gamma_i \circ \gamma_j^{-1} = \Psi\\) for now. We know we can differentiate on continuous functions, yet there are different degrees of differentiability as explained in  

[Differentiable manifolds](differentiable_manifolds.md)

It goes from \\(C^0\\) which is not differentiable but continuous to \\(C^{\infty}\\) which is infinitely differentiable.

Differentiable manifolds are therefore \\(C^1\\) and beyond. If nothing is stated, we assume that the transition functions are \\(C^{\infty}\\).

## Manifold learning?

So for my purpose, I am interested in how manifolds are used in machine and deep learning. 

I remember first learning about principal components and its generalization: singular value decomposition. 

It seemed extremely powerful: we can essentially “rid ourselves of the noise” by learning to map data onto a lower dimensional space. Of course, in terms of both these techniques we do: 

1. Assume linearity:
	We construct a lower dimensional space constructed of orthogonal vectors. Therefore we preserve linearity in the space. 
2. Only care about the global structure of the data:
	Furthermore, from this viewpoint, we look at the collective of data and find the best-fitting linear subspace. Because of this it only looks at the *global* structure of the data.

But life is not linear? 

In this sense, manifolds are a further generalization to PCA/SVD-techniques: IF a lower dimensional manifold existed that actually was linear in nature, then PCA and SVD are fine. More often that not however, this is not the case (if it is ever the case?).

Assuming that there exists a lower dimensional manifold, then, the question becomes: *Can we map the data we hape onto a lower dimensional manifold or at least approximate it?*

If yes, we would expect our model to learn the latent features ruling the patterns in the data. Thereby, we would end up with super generalizers for a given task.

So how do we reason that there exists a lower dimensional manifold?

This is a question that can be answered by various disciplines. Coming from a background in philosophy, (I am by no means a philosopher!), I like to think of the world as an entity that is given, not created. Therefore, we do not construct manifolds, they simply already exist.

- *This is tied to the [manifold hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis).*

The existence of manifolds. 

Following [Popper](https://en.wikipedia.org/wiki/Karl_Popper), we can never prove that a manifold exist, but we can attempt to disprove it. The great example is simply using PCA on a dataset, can we obtain useful information in a lower dimension? Yes? Then we have likely approximated a manifold.

Regardless of their philosophical status, manifolds are incredibly useful tools for understanding complex spaces, whether those are abstract mathematical spaces, the fabric of the universe, or in our case high-dimensional data sets.

So, our empirical evaluation is the yardstick for the degree of certainty of the existence of a **useful** manifold. Hence, whether we construct them or they already exist, it is really their respective utility we care about!
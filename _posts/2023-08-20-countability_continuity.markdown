---
layout: post
comments: true
title:  "Countability and continuity"
excerpt: "b;"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Countability and continuity

This relates very much to my previous notes on countability in set theory:

[Countability](../set_theory/countability.md)

## Countability

We have an open set in the topology \\(J\\)

If we use the open ball as base, that open set itself will also have an open set. 

In fact, there’s an infinite amount of open sets in an open set.

(as long as we dont have the discrete topology, there is going to be many if not an infinite number of open sets).

Are these sets countable?

For the open balls, the radius \\(r\\) used to draw the open balls are an element of the reals \\(r \in \mathbb{R}\\), wich is an infinite set. Therefore, open sets using the open balls as base are not countable.

However, we could always create a countable bunch. We could say, for example, that instead of drawing from the real numbers, we draw from the rational or even natural numbers.

If we can make countable open sets, nesting around a point \\(p \in X\\), what we are dealing with is open neighbourhoods! If we can do this for every point \\(x \in X\\), we say that the space is `first countable`.

Being `second countable` is a property of any open set. It essentially means, can we take any countable number of bases and unionise them to fully contain this open set. It turns out that for open balls as a basis, we can! (we can rely on \\(r \in \mathbb{Q}\\) and not \\(r \in \mathbb{I}\\).

First countable: the property in relation to points in the space.

Second countable: the property in relation to open sets in the space.

IF you are second countable, you are definitely  first countable, but not the other way around!

**Examples:**

- The Euclidian topology \\(J_E\\)  of \\(X \in \mathbb{R}\\) is both first and second countable, because:

	- we can make countable intervals around points in the line.
	- we can also unionise a countable number of intervals to fully contain the open set

- The nested interval topology is also first and second countable

- The cofinite topology (where we omit points along the line) is NOT first countable

	11:31 [https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4](https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4)

	![Untitled](/assets/manifold/assets/Untitled-6515262.png){: .zoom50% }

We want to have second countable properties when working with manifolds.

💡 We want manifolds to be second countable and Hausdorff, and the reason being is that manifolds are about things that are not dysfunctional, not weird strange sets.


Ultimately, we are looking for coordinates we can do calculus on, we can take derivatives. We need these properties to be able to do this.



## Continuity

We have a function between the domain and the range. In this case, the domains are not just sets, but topological spaces. In this sense, the domain is endowed with the topology and the range is enowed with the topology. This is what allows us to define continuity. Otherwise, if the sets were not endowed in a topology, it was simply a list of points from the domain corresponding to points in the range.

```mermaid
graph LR
	a((X, Jx))
	b((Y, Jx))
	a ==.f.==> b;
	b ==.f^-1.==>a;
```

There’s also an inverse function (really a mapping), from points in the range to points in the domain. 

The video shows that the function \\(f\\) takes points from the domain and maps them to points in the range. 

The video shows that it is not necessarily the case for the inverse \\(f^{-1}\\). Here, a point in the range be mapped to two different points in the codomain. 

I suspect the reason being is that the range is of lower dimensionality?

Definition

If we have a point \\(x\\) in the domain, and a point \\(y\\) in the range. 

A function is continuous if any open set in \\((Y, J_Y)\\), if the pre-image under the inverse is an open set in \\((X, J_X)\\), then \\(f\\) is continuous.

So \\(f\\) is continuous, it depends on any open set \\(u \in J_y\\) and the inverse mapping of \\(u\\), \\(f^{-1}(u)\\). The inverse mapping has to be an element in the topology of \\(X\\). 

\\(\\)
f^{-1}(u) \in J_X
\\(\\)

In terms of neighbourhoods:

If \\(f(x)\\) is a point in \\(Y\\), and \\(x\\) is a point in \\(X\\). And \\(u \in J_X\\) is an open set around \\(x\\) and \\(v \in J_Y\\) is an open set around \\(y\\). I need to be able to find an open neighbourhood such that the image \\(u\\) is a subset of \\(v\\): \\(f(u) \subset v\\).

So for every \\(v\\) that is an open neighbourhood of \\(f(x)\\), for the function to be continuous, I need to be able to find an open neighbourhood in \\(x\\) with that entire open neighbourhood maps to a subset of the open neighbourhood in \\(v\\).

![Untitled](/assets/manifold/assets/Untitled 1.png){: .zoom50% }

**Examples:**

The euclidian line, 34:00 in [https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4](https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4)

When you are dealing with manifolds, you are dealing with continuous functions from one manifold to another. The topologies in the manifolds are not the weird topologies shown in the video, they are all much more like the Euclidian topology. They are robust to capture everything we need, they are continuous and we are not dealing with mapping from one weird topology to another.

## Homeomorphisms

[Wolfram](https://mathworld.wolfram.com/Homeomorphism.html)

> A homeomorphism, also called a continuous transformation, is an [equivalence relation](https://mathworld.wolfram.com/EquivalenceRelation.html) and [one-to-one correspondence](https://mathworld.wolfram.com/One-to-OneCorrespondence.html) between points in two geometric figures or [topological spaces](https://mathworld.wolfram.com/TopologicalSpace.html) that is [continuous](https://mathworld.wolfram.com/Continuous.html) in both directions. A homeomorphism which also preserves distances is called an [isometry](https://mathworld.wolfram.com/Isometry.html). [Affine transformations](https://mathworld.wolfram.com/AffineTransformation.html) are another type of common geometric homeomorphism.

[https://en.wikipedia.org/wiki/Homeomorphism](https://en.wikipedia.org/wiki/Homeomorphism)

IF \\(f\\) is a function, and \\(f^{-1}\\) is a function, and both are continuous, then we are dealing with a homeomorphism.

\\(\left(f^{-1}\right)^{-1}=f\\)

Properties of topological spaces:

- Compactness
- Separability
- Connectedness

The homeomorphism ensures that the properties are “carried over” from the domain to the range and vice versa.
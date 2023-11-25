---
layout: post
comments: true
title:  "Differentiable manifolds"
excerpt: "An insightful exploration of Differentiable manifolds"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Differentiable manifolds

TS → TM → DM

All differentiable manifolds are topological manifolds.

Not all topological manifolds are differentiable manifolds.

All topological manifolds are topological spaces.

Not all topologial spaces are topological manifolds.

What is the distinction between DM and TM?

**The topological manifold**: \\((X, \mathcal{T}_X, \mathcal{A})\\), where the atlas contains charts \\(\mathcal{A} = \{(U_i, \gamma_i)\}\\)

All of these charts cover the entire topological space \\(X\\), which forms the atlas. Locally each chart cover an open neighborhood which is an open subset of the topology. And each chart region is locally homeomorphic to some region of the euclidean space \\(\mathbb{R}^d = \mathbb{R} \times \mathbb{R} \times \dots \times \mathbb{R}_d\\) where \\(\times\\) is the cartesian product, which yields ordered pairs that can be used as coordinates in the euclidean space.

![Untitled](/assets/manifold/assets/Untitled-6515497.png){: .zoom50% }

We can do calculus in the euclidean space!

If we take a single point \\(p\\), which \\(U_1\\) centers around, it is essentially a large d-dimensional tuple, which is the coordinates of \\(p\\) in the euclidean space: \\(\gamma_1(p)=\{\alpha^1,\alpha^2,\dots, \alpha^d \}\\). The coordinates only makes sense in that specific chart \\((U_1, \gamma_1)\\). \\((U_2, \gamma_2)\\) has its own set of coordinates and are NOT comparable to other charts, because the charts are different euclidian spaces of the same dimensionality.

**What criteria must be made to make these topological manifolds differentiable?**

We need to understand transition functions, those transition functions are going to be maps to the different charts (when appropriate!).

## Transition functions

If we have an atlas \\(\mathcal{A} = \{ (U_1, \gamma_1), (U_2, \gamma_2), \dots \}\\) then we can look at the two open sets as \\(U_{1,2} = U_1 \cap U_2\\). \\(U_{1,2}\\) is a [simply connected](https://mathworld.wolfram.com/SimplyConnected.html) space and \\(\gamma\\)’s are homeomorphisms: 1-1, onto, continuous, the inverse is continuous. This means, they will preserve topological properties which means that the images keep these properties.

The sphere example from earlier showed a stereographic function, that overlapped almost entirely between the two sets aside from two points, the north and the south pole. 

[Topological manifolds](topological_manifolds.md)

Charts have overlaps, however, we do not need extreme overlaps.

Any point \\(p \in U_{1,2}\\), can be mapped into the regions of the euclidean space by either \\(\gamma_1(p)\\) or \\(\gamma_2(p)\\)

All the points in \\(U_{1,2}\\) will thus be mapped to some sub-region of \\(\gamma_1(U_1)\\) and \\(\gamma_2(U_2)\\).

We want to make a map from one euclidean sub-region of one chart to another euclidean subregion of another chart (\\(\mathbb{R}^d \rightarrow \mathbb{R}^d\\)). How do we build this map?

Let's say we want to map the point \\(p\\) from the euclidean range, obtained via \\(\gamma_1(p)\\). We simply need to leverage the fact that \\(\gamma\\) is bi-continuous, so we can easily map it back to the domain. Then, once we have \\(p\\) in the domain, we can map it to the euclidean range via \\(\gamma_2(p)\\):

\\(\\)
\gamma_2\left(\gamma_1^{-1}\left(\gamma_1(p)\right)\right)
\\(\\)

This looks like the standard functional notation. We don’t like this notation.

Instead we say \\(\gamma_2 \circ \gamma_1^{-1}(\gamma_1(p))\\), where \\(\gamma_1(p)\\) is the coordinates of p in the euclidean range, and \\(\gamma_2 \circ \gamma_1^{-1}\\) is simply a composed function that eats the coordinates of the point in the euclidean range.

**This is the transition function!**

It transitions from one chart to another! and we can take derivatives of this, because it maps from \\(\mathbb{R}^d \rightarrow \mathbb{R}^d\\)

![Untitled](/assets/manifold/assets/Untitled 1-6515493.png){: .zoom50% }

These functions can be made given we have a topological manifold. Because the composition of the two functions \\(\gamma\\) are continuous, \\(\gamma_i \circ \gamma_j^{-1}\\) is also continuous. Given we have a topological manifold, we are guarranteed that the composition functions are continuous. This means they are at least \\(C^0\\). 

If we like to take derivatives, we need something stronger than just \\(C^0\\). We need \\(C^1\\).

What does this mean?

- \\(C^1\\) means that the first derivative exists and is continuous.
	- This allows you to perform basic calculus operations like taking gradients.
- \\(C^2\\) means that the second derivative exists and is continuous.
	- This is useful for understanding curvature, acceleration, etc.
- … You get the pattern.
- \\(C^{\infty}\\) Functions are smooth.
	- meaning they have continuous derivatives of all orders. This is often the most desirable case because it allows for the most extensive calculus operations

We are interested in functions that are \\(C^{\infty}\\)

To have \\(C^{\infty}\\) means that the function is “**smooth**” because it's the most "permissive" level of smoothness, allowing you to do pretty much any calculus operation you'd like without having to worry about the function's behavior. It's the mathematical equivalent of having a Swiss Army knife with an infinite number of tools.

Example: a simple linear regression (\\(f(x) = ax + b\\) ) is \\(C^{\infty}\\)

1. **First Derivative**: *f*′(*x*)=*a* (constant)
2. **Second Derivative**: *f*′′(*x*)=0 (constant)
3. **Higher-Order Derivatives**: *f*(*n*)(*x*)=0 for *n*≥2 (all zero, hence constant)

All these derivatives exist and are continuous, making \\(f(x)=ax+b\\) a \\(C^{\infty}\\) function.


💡 ***IF THE TRANSITION FUNCTIONS ARE SMOOTH, WE HAVE A DIFFERENTIABLE MANIFOLD!**


This is the perferct differentiable manifold, however some loosen the criteria and says that it is a \\(C^k\\)-differentiable manifold for \\(k \geq 1\\).

## Example of transition functions: Coordinates

We go back to the sphere and we want to map it to \\(\mathbb{R}^2\\) with the usual topology.

![Untitled](/assets/manifold/assets/Untitled 2-6515488.png){: .zoom50% }

In principle, the R2 is different from \\(\gamma_1(p)\\) and  \\(\gamma_2(p)\\), but we use it for illustrational ease. 

So it is clear that the composite function maps directly from the coordinate plane R2 to another coordinate plane R2.
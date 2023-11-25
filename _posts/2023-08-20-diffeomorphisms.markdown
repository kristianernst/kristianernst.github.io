---
layout: post
comments: true
title:  "Diffeomorphisms"
excerpt: "“Diffeomorphisms is the homeomorphisms for differentiable manifolds”"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Diffeomorphisms

## The curve

We define the curve as a mapping of \\(\lambda\\) to the differential manifold \\(X\\). 

Let \\(\lambda\\) be any point on an open line, \\(\lambda \in \mathbb{R}\\). 

Let the function \\(f\\) map \\(\lambda\\) to X,  \\(f(\lambda) \in X\\) because \\(f(\lambda): \mathbb{R} \rightarrow X\\)

![Untitled](/assets/manifold/assets/Untitled-6515580.png){: .zoom50% }

This picture tells that to construct curves and use them  for calculus, we can map then onto the differential manifold and then map them to a euclidean range, in this case \\(\mathbb{R}^2\\) with the usual topology. Hence to optain the coordinates in the chart, \\((a,b) = \gamma \circ f(\lambda_i) \in \mathbb{R}^2\\), for a given value of \\(\lambda\\). We really need the chart \\(\gamma\\) for this whole thing to work, otherwise we would not be able to obtain coordinates. 

Remember a chart is \\((U_i, \gamma_i)\\) (an open set / neighborhood around a point i and a continuous function mapping to the euclidean space.)

## Coordinate functions

If we have a chart \\((U, \gamma)\in \mathcal{A}\\), and a topological manifold: \\((X, \mathcal{T}_X, \mathcal{A})\\). 

We want to map a point \\(p\\) in the open cover / chart onto the euclidean space that has a usual topology. 

We know this can be done using \\(\gamma(p) = (\alpha^1, \alpha^2) =(a,b)\\), given that \\(\mathbb{R}^2\\).

We could also view \\(\gamma(p) = \left(\alpha^1(p), \alpha^2(p)\right)\\), here the \\(\alpha\\)s are coordinate functions. Each coordinate function takes a point \\(p \in U\\), \\(\alpha^i(p): U \Rightarrow \mathbb{R}\\). 

\\(\gamma\\) is then the functions that collects these coordinate functions together and construct a higher dimensional euclidian space.

## General relativity reference

![Untitled](/assets/manifold/assets/Untitled 1-6515584.png){: .zoom50% }

This picture shows how a new coordinate \\(X^{0^{\prime} }\\) is a function of old coordinates. 

\\(X^{0^{\prime} } (X^0, X^1)\\), The \\((X^0, X^1)\\) is the coordinates in the plane \\(\mathbb{R}^2\\) which is equivalent to \\(\gamma(p)\\). The \\(X^{0^{\prime} }()\\) then is a coordinate function that returns a real number which is just the \\(X^{0^{\prime} }\\) coordinate. In this sense, \\(X^{0^{\prime} } () = X^{0^{\prime} }\circ \gamma^{-1}\\). Combining them, we have the familiar expression: \\(X^{0^{\prime} } \circ \gamma^{-1} \left(\gamma(p)\right)\\)

The function \\(X^{0^{\prime} }\\) and the coordinate \\(X^{0^{\prime} }\\) is used redundantly in GR theory.

Of course, we can do the same for the coordinate \\(X^{1^{\prime} }\\).

Similarly, the old coordinates can be expressed as a function of the new coordinates:

\\(X^0 = X^0 \circ \phi^{-1}\left(\phi(p)\right)\\).

## Functions from one manifold to another

Is \\(f\\) a differentiable function?

\\(f : X \rightarrow Y\\), where \\(X\\) is a differentiable manifold \\((X, \mathcal{T}_X, \mathcal{A})\\), and \\(Y\\) is a differentiable manifold \\((Y,\mathcal{T}_Y,\mathcal{B})\\)

![Untitled](/assets/manifold/assets/Untitled 2-6515589.png){: .zoom50% }

Instead of using transition functions from one chart to another of an atlas, can we transition from one manifold to another? (this is marked with the purple line)

IF its differentiable everywhere, not just in the intersections, \\(f\\) is called a smooth differentiable function between the two manifolds. 

And as such, once again in order to understand the differentiability, we have to lean on all the structures of the manifolds. We need to lean on the charts, the chart maps \\(\gamma\\), the regions of the chart maps \\(U\\)

Just because \\(f\\) is differentiable, it does not necessarily mean that it is one-to-one and onto, and the inverse of \\(f\\) is also differentiable in the same way as \\(f\\) is. 


💡 However, IF this is the case that \\(f\\) is all of these things (basically everything in a homeomorphism and differentiability) then the two spaces are `diffeomorphic`.

> “Diffeomorphisms is the homeomorphisms for differentiable manifolds”
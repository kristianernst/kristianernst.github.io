---
layout: post
comments: true
title:  "Topological manifolds"
excerpt: "A topological space \\(M\\) satisfying some separability (i.e. it is a T2 space) and countability (i.e. it is paracompact) conditions such that every point \\(p \in M\\) has a neighbourhood homeomorphic to an open set in \\(\mathbb{R}^n\\) for some \\(n \geq 0\\). Every smooth manifold is a topological manifold but not necessarily vice versa. "
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Topological manifolds

Definition: 

> A topological space \\(M\\) satisfying some separability (i.e. it is a T2 space) and countability (i.e. it is paracompact) conditions such that every point \\(p \in M\\) has a neighbourhood homeomorphic to an open set in \\(\mathbb{R}^n\\) for some \\(n \geq 0\\). Every smooth manifold is a topological manifold but not necessarily vice versa. 
> …
> For manifolds, Hausdorff and second countable are equivalent to Hausdorff and paracompact, and both are equivalent to the manifold being embeddable in some large-dimensional Euclidean space.

We have a topological space \\((X, T_X)\\) with the properties:

- Hausdorff
- 2-C
- Paracompactness

We also assume the notion of “locally euclidian”. 

The trick is that we can view open sets of the topology of X as individual topological spaces themselves.

If we can create a function for any open set in the topology of X, that is homeomorphic to a topological space \\((Y , T_Y)\\) where \\(T_Y\\) is the usual topology in a euclidian space: \\(Y \in \mathbb{R}^n\\).

In this sense, we have created a cover of neighbourhoods that cover all points in \\(X\\), and we need to show that all these neighbourhoods has a mapping to \\(Y\\) that is homeomorphic. If we can do this, we have shown that the space is `locally euclidian`

**Example of mapping to R2:**

![Untitled](/assets/manifold/assets/Untitled-6515426.png){: .zoom50% }

### Example: Sphere

Consider a sphere

the set we are interested in are the points on the surface of the sphere.

the topological space is: \\(\{X,T_{\mathbb{R}^3|X} \}\\)

What does this mean? 

- Well \\(X\\) is the set of points of the surface of the ball. Not the interior nor the exterior.
- This set X needs a topology, which is \\(\mathbb{R}^3\\). \\(\mathbb{R}^3\\) has a basis of open balls of spherical form. Some of these open balls will intersect with the surface of the sphere as described by X, which will result in disc shaped objects on the surface of the sphere. These are part of the basis for the topology of the sphere.
	- The null set is part of the topology for open balls not intersecting the sphere
	- The sphere itself is part of the topology for the open balls that fully contain the sphere.

Can we map points on the sphere to a topological space \\(\mathbb{R}^2\\)?

- We know the sphere is compact, i.e. there is a finite sub cover for any open cover of the sphere.
- We know the plane \\(\mathbb{R}^2\\) is NOT compact.
- Because of 1 and 2, we also know that there is no homeomorphic function that maps the sphere to the plane. Because compactness is a topological property.
- HOWEVER, if we just take ONE point of the sphere, the sphere becomes non-compact.
	- All of a sudden, the sphere is homeomorphic to the plane.

Why the sphere is compact:

The surface of the sphere is the set of all points \\((x,y,z) \in \mathbb{R}^3\\) that satisfy the equation \\(x^2 + y^2 + z^2 = r^2\\). 

1. **Closed**: The surface of the sphere is a closed set because it contains all its boundary points. In fact, the surface itself acts as its own boundary in \\(\mathbb{R}^3\\).
2. **Bounded**: The surface is also bounded because you can enclose it in a larger ball of radius \\(r+ϵ\\) for some \\(ϵ>0\\).

By the [Heine-Borel theorem](https://en.wikipedia.org/wiki/Heine–Borel_theorem), since both properties of the subset is true, it is compact.

Cont:

On the other hand, every spot is locally homeomorphic to a subset of the euclidian space. 

The mapping is not between the sphere and the plane, it is between the topological space open neighborhood of a point p with the subspace topology to a topological space wich is an open neighborhood of the plane \\(\mathbb{R}^2\\) with its subspace topology.

**Chat GPT**

- If you take an open neighborhood around a point *p* on the sphere, you can indeed find a homeomorphism between that neighborhood and an open set in R2. This is a fundamental idea in differential geometry and topology, often discussed under the topic of "charts" and "atlases."

	if we have a point \\(p\\) on the surface of the sphere \\(S^2\\) in \\(\mathbb{R}^3\\), (S2 because it is simply the boundary, i.e. no volume), you can consider an open neighborhood \\(U\\) around \\(p\\) that lies entirely on \\(S^2\\).

	We need a homeomorphic mapping \\(U\\) to an open set \\(V \in \mathbb{R}^2\\). 

	The mapping \\(f:U\rightarrow V\\) will be a homeomorphism if it is:

	1. **Bijective**: A one to one and onto mapping between \\(U\\) and \\(V\\)
	2. **Continuous**: The map \\(f\\) is continuous
	3. **Inverse is continuous**: the inverse map \\(f^{-1}: V \rightarrow U\\) is also continuous

	For small enough neighborhoods, these conditions can be met, making \\(f\\) a homeomorphism between \\(U\\) and \\(V\\).

	The key idea is that you are looking at a local piece of the sphere, not the entire sphere. Many global properties (like compactness) prevent a homeomorphism between the whole sphere and the plane. These don’t come into play, when you look at small open neighbourhoods.

lets construct a neighborhood \\(U_p = X \setminus q\\), where \\(q\\) is simply a point on the sphere.

Here's how a geometric interpretation of a function \\(f\\) mapping \\(U_p\\) to \\(V\\)

![Untitled](/assets/manifold/assets/Untitled 1-6515433.png){: .zoom50% }

Then we do it again, but instead of removing the south pole, we remove the north pole.

\\(U_q = X \setminus p\\)

![Untitled](/assets/manifold/assets/Untitled 2-6515436.png){: .zoom50% }

We then take these two homeomorphisms, pair them up with the sets they are attached to, and put them in a set together

\\(\{ (U_p, f_p), (U_q, f_q)\}\\), the () () are individually called charts, and the set containing all the `charts`, covering the entire topological space in question is called the `atlas`

### Counter example: Line with an orthogonal line

So we want to map \\((X, T_{\mathbb{R}^2 | X})\\) to \\(\mathbb{R}^1\\)

And X is essentially two lines, where one begins from a point on the other line and is ortogonal to that line.

![Untitled](/assets/manifold/assets/Untitled 3-6515441.png){: .zoom50% }

It is impossible to construct a homeomorphic relationship between these two topological spaces. It is possible to map from the higher dimensional space to the lower dimensional one, but not the other way around. Therefore \\(f^{-1}\\) is not onto and one to one (and is thus not invertible).
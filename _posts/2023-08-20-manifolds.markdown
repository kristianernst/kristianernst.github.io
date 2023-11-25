---
layout: post
comments: true
title:  "Manifolds"
excerpt: "An insightful exploration of Compactness, Connectedness and Topological properties"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

## General note:

These notes was based on the youtube video [series](https://www.youtube.com/watch?v=CEXSSz0gZI4&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo) by XylyXylyX



- [General note:](#general-note)
- [Topology](#topology)
  - [What is an open set?](#what-is-an-open-set)
  - [What is a closed set?](#what-is-a-closed-set)
  - [The notion of limit points](#the-notion-of-limit-points)
    - [Open neighbourhood](#open-neighbourhood)
    - [Limit points](#limit-points)
  - [Closure](#closure)
  - [Interior](#interior)
  - [Exterior](#exterior)
  - [The Boundary](#the-boundary)
  - [Density](#density)
- [Separability](#separability)
  - [The Base of the topology](#the-base-of-the-topology)
  - [Separability](#separability-1)
- [Topological manifolds](#topological-manifolds)
  - [Example: Sphere](#example-sphere)
  - [Counter example: Line with an orthogonal line](#counter-example-line-with-an-orthogonal-line)
- [Compactness, Connectedness and Topological properties](#compactness-connectedness-and-topological-properties)
  - [Compactness](#compactness)
    - [Covers, closed and open sets](#covers-closed-and-open-sets)
    - [Back to the example](#back-to-the-example)
  - [Connectedness](#connectedness)
  - [Homotopy](#homotopy)
- [Countability and continuity](#countability-and-continuity)
  - [Countability](#countability)
  - [Continuity](#continuity)
  - [Homeomorphisms](#homeomorphisms)
- [Diffeomorphisms](#diffeomorphisms)
  - [The curve](#the-curve)
  - [Coordinate functions](#coordinate-functions)
  - [General relativity reference](#general-relativity-reference)
- [Functions from one manifold to another](#functions-from-one-manifold-to-another)
- [Differentiable manifolds](#differentiable-manifolds)
  - [Transition functions](#transition-functions)
  - [Example of transition functions: Coordinates](#example-of-transition-functions-coordinates)
- [Recap 1: Constructing manifolds?](#recap-1-constructing-manifolds)
  - [Quick on topological manifolds](#quick-on-topological-manifolds)
  - [Quick on differentiable manifolds](#quick-on-differentiable-manifolds)
  - [Manifold learning?](#manifold-learning)



## Topology

Point set topology / general topology is the study of the general abstract nature of continuity or "closeness" on [spaces](https://mathworld.wolfram.com/Space.html). Basic point-set topological notions are ones like [continuity](https://mathworld.wolfram.com/ContinuousSpace.html), [dimension](https://mathworld.wolfram.com/Dimension.html), [compactness](https://mathworld.wolfram.com/CompactSpace.html), and [connectedness](https://mathworld.wolfram.com/ConnectedSpace.html).

If we have a set of points \\(X\\), \\(J_X\\) is a topology of \\(X\\) that contains a lot of subsets of \\(X\\).

These subsets inside the topology are called open sets \\(u\\). 

$$
u_1, u_2, u_3 \in J_X
$$

Rules:

$$
\begin{align}\cup u_i \in J_X \end{align}
$$

$$
\begin{align} \cap u_i \in J_x \end{align}
$$

The intersections have to be a finite number of open sets.

The null set is also in the topology of X and the set itself is an element in the topology of X. These mark the smallest and largest elements of X.

$$
\emptyset \in J_X, \quad X \in J_X
$$

### What is an open set?

We use the notion “[open balls](https://mathworld.wolfram.com/OpenBall.html)” to constitute open sets. 

It is understood by reference to a euclidean n-space. For example, if we are in a two-dimensional euclidian space, the open ball is seen as a disc. On the contrary, if we are in 3D, the open ball is seen as a sphere. Note however, the “surface” of the ball is not contained by the set that constitutes the open ball.

The open balls become the `base` of the topology, because we can union an infinite amount sets constructed by open balls to make up the topology \\(J_X\\). 

We could also use open rectangles or another shape for that matter, but the common practice is open balls.

There exists different types of topologies:

1. The usual topology is the topology constructed by some combination of open balls.
2. The discrete: which amounts to the power set \\(2^X\\), i.e. it contains all possible sets of \\(X\\).
3. `The trivial topology` which amounts to \\(J_X = \{ \emptyset, X\}\\), simply contains \\(X\\) itself and the empty set.

### What is a closed set?

A closed set is simply the complement of an open set.

\\(X \setminus u\\)

The rules following a closed set is the almost the same as (1) and (2), except now we can intersect over an infinite number of closed sets and unionise a finite number of closed sets.

### The notion of limit points

#### Open neighbourhood

An open neighbourhood of a point P is an open set that contains the point P

Therefore, open sets are open neighbourhoods of all of its points.

#### Limit points

Let \\(P\\) be a limit point of \\(S\\), where \\(S \subset X\\), IF every neighbourhood of P (\\(u_{P}\\)) contain some element of \\(S\\).

Thus the following must be true:

$$
u_P\cap S \neq \emptyset
$$

The following is NOT always true:

$$
u_P \cap S = u_p \ ,
$$

because, \\(u_P\\) can exist on the border of \\(S\\), which still constitutes a limit point

The same is true for a single missing point inside of \\(S\\).  

### Closure

The closure of the set is the set unionised with all of its limit points. 

- If you have an open ball, the closure of that ball is the open ball plus its boundary
	- [ClosedDisk](https://mathworld.wolfram.com/ClosedDisk.html)

### Interior

A point \\(P\\) is in the interior of \\(S\\) if I can find an open set of \\(P\\) \\(u_P\\) that is a subset of \\(S\\).


$$
u_P \cap S = u_p \  \rightarrow p\in S^{o}
$$

**Important:** The interior of any set is always open!

The union of every open set contained in S is the interior of S. 

If S is an open set, \\(S = S^{o}\\)

The interior is the complement of the closure of the complement of \\(S\\).

### Exterior

The exterior of a set \\(S\\) is the complement of its closure. 

[video](https://www.youtube.com/watch?v=EsF_5LoaL_8&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=2) 27:48

### The Boundary

What is not in the exterior nor in the interior.


<img src="/assets/manifold/Untitled.png" alt="Screenang" style="zoom: 33%;" />

### Density

\\(S\\) is dense in \\(X\\), if every point in \\(X\\) (\\(P \in X\\))  is either also a point in \\(S\\) or a limit point of \\(S\\). Hence we can simplify it to say that \\(P \in \bar{S}\\) where \\(\bar{S}\\) is the closure of \\(S\\)


## Separability

### The Base of the topology

We use a Line as examples:

Different options:

- the Euclidean topology, simply the line
	- the usual line (open ball topology)
- The nested interval: (0, 1 - 1/n)
	- nested line (same set but with a different topology!)
- closed interval topology: X is the set (-1, 1)
	- base constituted of half open intervals: [-1, a) (b,1] where a > 0 and b < 0.

### Separability

\\(T_0\\) separability:

For any two points \\(x, y\\) there is an open set \\(U\\) such that \\(x \in U\\) and \\(y \notin U\\) or \\(y \in U\\) and \\(x \notin U\\). A space fulfilling this axiom is called a T0-space.

It is easy in the euclidean topology, where you just take an interval that only includes one point.

\\(T_1\\) separability; frechet 

For any two points \\(x, y \in X\\) there exists two open sets \\(U\\) and \\(V\\) such that \\(x \in U, y \notin U\\) and \\(y \in V, x \notin V\\). A space satisfying this axiom is known as a [T1-space](https://mathworld.wolfram.com/T1-Space.html)

\\(T_2\\) separability; Hausdorff (This is the one you usually encounter!)

For any two points \\(x, y \in X\\) there exists two open sets \\(U, V\\) such that \\(x \in U, y \notin U\\) and \\(y\in V, x \notin V\\), and \\(U \cap V = \emptyset\\)

This is an extension of T1 that also says that the two sets V and U has to be disjoint!


<img src="/assets/manifold/Untitled-6515138.png" alt="as" style="zoom: 33%;" />

It is clear that if \\(T_1\\) is true, then \\(T_0\\) is automatically true, however not the other way around.

Here is the complete separability axiom ranking:

\\(T_4 \Rightarrow T_3  \Rightarrow T_2  \Rightarrow T_1  \Rightarrow T_0\\)


## Topological manifolds

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

<img src="/assets/manifold/Untitled-6515426.png" alt="as" style="zoom: 33%;" />

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

<img src="/assets/manifold/Untitled-6515433.png" alt="ddd" style="zoom: 33%;" />
Then we do it again, but instead of removing the south pole, we remove the north pole.

\\(U_q = X \setminus p\\)

<img src="/assets/manifold/Untitled-6515436.png" alt="222" style="zoom: 33%;" />

We then take these two homeomorphisms, pair them up with the sets they are attached to, and put them in a set together

\\(\{ (U_p, f_p), (U_q, f_q)\}\\), the () () are individually called charts, and the set containing all the `charts`, covering the entire topological space in question is called the `atlas`

### Counter example: Line with an orthogonal line

So we want to map \\((X, T_{\mathbb{R}^2 | X})\\) to \\(\mathbb{R}^1\\)

And X is essentially two lines, where one begins from a point on the other line and is ortogonal to that line.

<img src="/assets/manifold/Untitled-6515441.png" alt="333" style="zoom: 33%;" />
It is impossible to construct a homeomorphic relationship between these two topological spaces. It is possible to map from the higher dimensional space to the lower dimensional one, but not the other way around. Therefore \\(f^{-1}\\) is not onto and one to one (and is thus not invertible).


## Compactness, Connectedness and Topological properties

<img src="/assets/manifold/Untitled-6515326.png" alt="555" style="zoom: 33%;" />

How do we know that a space is homeomorphic to another space? (in this case X and Y). Any single function that is 1-1 onto and i continuous and has a continuous inverse, then the two spaces are homeomorphic.

It does not matter that ALL functions, we only need ONE function to achieve this.

Once we have achieved this, then every topological property of one space is also true in the other space homeomorphic to it.

By the same token, if we know that two spaces do not have the same properties, we KNOW that there exists no homeomorphic function. fx. if X is second countable and Y is not.

Open sets are matched to open sets in both directions. I take every element in the topology and I am transforming/mutating it. These homeomorphic functions are transforming/morphing them (we change the shape of things).

A doughnut = a coffee cup! They are homeomorphic. every open set on the doughnut can be mutated into some open set of the coffee cup and vice versa.

Homeomorphsim is the weakest notion that connects two spaces. There are no notions of distance or geometry or anything else. Topology is the geometry without all the measurements.

Properties:

- Countability: inf and finite, etc.
- Separability: T1, T2, T3, REG, etc…
- Compactness:
- Connectedness:

### Compactness

Definition: A subset \\(S\\) of a topological space \\(X\\) is compact if for every open cover, there exists a finite subcover of \\(S\\).

Is the topological space \\((X, J_X)\\) compact?

First, let us ask about a subset. let the open set \\(A \subset (X, J_X)\\), is \\(A\\) compact?

#### Covers, closed and open sets

An open cover of a subset \\(S\\) of \\(X\\) is a collection \\(\mathcal{C} = \{ U_\alpha \}\\) of open sets in \\(X\\) such that 

$$
S \subseteq \bigcup_\alpha U_{\alpha}
$$

A subcover is a sub collection \\(\mathcal{C}^{\prime} \subseteq \mathcal{C}\\) that still covers \\(S\\).

A subset \\(S\\) of \\(X\\) is compact if every open cover \\(\mathcal{C}\\) of \\(S\\) has a finite subcover.

An open set in a topological space \\(X\\) is a set \\(U\\) such that for every point \\(x \in U\\) there exists an open neighbourhood \\(N\\) of \\(x\\) such that \\(N\subseteq U\\). In simpler terms, this means that you can wiggle around any point in the set without ever stepping out of it.

A closed set in a topological space \\(X\\) is a set \\(C\\) such that its complement \\(X \setminus C\\) is open. This means that a closed set contains all its `boundary points`. A boundary point of \\(S\\) is a point \\(x\\) such that every open neighborhood of \\(x\\) intersects both \\(S\\) and \\(X \setminus S\\). 

1. **Open Sets and Compactness**: Open sets don't contain their boundary points. When you try to cover an open set with smaller open sets (an open cover), you'll find that you can't pin down the boundary with a finite number of them. You'll always need "just one more" to get closer to the boundary, leading to an infinite subcover.
2. **Closed Sets and Compactness**: Closed sets do contain their boundary points. When you cover a closed (and bounded) set with open sets, you can always find a finite subcover that includes the boundary. This makes the set compact, at least in Euclidean spaces.

#### Back to the example

The number of open sets we need to cover \\(A\\) is 1. Obviously we can cover \\(A\\) with \\(1, \dots, \infty\\)

We can use infinite open sets in two ways:

- Make a cover with infinite amount of open sets to cover the subset. (an infinite cover with no finite sub-cover)
- Make an cover with infinite amount of open sets that cover the whole space, thereby ultimately cover the subset with some finite number of these open sets. (an infinite cover with finite sub-cover)

Example: 

The video uses an example of a rectangle in the usual topology of the plane. It compares an open rectangle in which you can fill up the interior of the rectangle (which is equal to the open rectangle itself, by an infinite amount of open subsets. This is then the cover. For this cover, therefore, there is no finite subset, since it would then not be a cover.

Contrary, the closed rectangle. By a similar fashion we can begin to cover the rectangle by an infinite amount of open sets. Here we get infinitely close to the border of the rectangle. However, by this token, we cannot reach the border and thus we need an additional cover to cover the rectangle. Because of this, there will exist a finite sub-cover for the open cover. 

### Connectedness

Definition: A connected set is a set that cannot be partitioned into two non-empty subsets which are open in the relative topology induced on the set. Equivalently, it is a set which cannot be partitioned into two non-empty subsets such that each subset has no points in common with the set closure of the other.

If we have a topological space \\((X, J_X)\\) 

<img src="/assets/manifold/Untitled-6515321.png" alt="545" style="zoom: 33%;" />

Where the space X consists of two spaces G and H, let G and H be open sets in the topology, if there is no intersection between them, then we can say that the space \\(X\\) is NOT connected.

By the property of homeomorphism, if we map to another topological space, the property of connectedness follows:

d<img src="/assets/manifold/Untitled 2.png" alt="546" style="zoom: 33%;" />

**Path-connectedness**

[https://mathworld.wolfram.com/Pathwise-Connected.html](https://mathworld.wolfram.com/Pathwise-Connected.html)

Path-connectedness is another way of understanding connectedness

If we have a topological space \\(X \in \mathbb{R}^2\\)

and we have a continuous function \\(f\\) from \\([0,1]\\) to \\(X\\) such that \\(f(0) = x\\) and \\(f(1)=y\\). 

If you are path connected, you are connected.


<img src="/assets/manifold/Untitled 3.png" alt="526" style="zoom: 33%;" />


### Homotopy

Definition: A continuous transformation from one function to another. A homotopy between functions \\(f\\) and \\(g\\) from a space \\(X\\) to a space \\(Y\\) is a continuous map \\(G\\) from \\(X \times [0,1] \rightarrow Y\\) such that \\(G(x,0)=f(x)\\) and \\(G(x,1)=g(x)\\) where \\(\times\\) denotes set pairing. another way of saying this is that a homotopy is a path in the mapping space \\(\operatorname{Map}(X,Y)\\) from the first function to the second. Two mathematical objects are said to be [homotopic](https://mathworld.wolfram.com/Homotopic.html) if one can be continuously deformed into the other.

<img src="/assets/manifold/Untitled 4.png" alt="527" style="zoom: 33%;" />

The doughnut have different homotopies of paths because of the hole in the space. I.e. the blue function of path-connectedness behaves vastly different from that of the red line, and the green line and vice versa. However in the topological space of \\(X\\), the red line and the blue line and the green line can be deformed to the other (this is called “simply connected” \\(C \sim X\\)).

The properties:

- separability
- countability
- compactness
- connectedness

Are all part of the topological space. That are carried over via homeomorphism, that allow us to deform.

We do not have notions of shape, distance and other measurements.

## Countability and continuity

This relates very much to my previous notes on countability in set theory:

[Countability](https://ernst-hub.github.io/math/2023/08/20/set_theory/)

### Countability

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

	11:31 [youtube](https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4)

<img src="/assets/manifold/Untitled-6515262.png" alt="520" style="zoom: 33%;" />

We want to have second countable properties when working with manifolds.

💡 We want manifolds to be second countable and Hausdorff, and the reason being is that manifolds are about things that are not dysfunctional, not weird strange sets.


Ultimately, we are looking for coordinates we can do calculus on, we can take derivatives. We need these properties to be able to do this.

### Continuity

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

$$
f^{-1}(u) \in J_X
$$

In terms of neighbourhoods:

If \\(f(x)\\) is a point in \\(Y\\), and \\(x\\) is a point in \\(X\\). And \\(u \in J_X\\) is an open set around \\(x\\) and \\(v \in J_Y\\) is an open set around \\(y\\). I need to be able to find an open neighbourhood such that the image \\(u\\) is a subset of \\(v\\): \\(f(u) \subset v\\).

So for every \\(v\\) that is an open neighbourhood of \\(f(x)\\), for the function to be continuous, I need to be able to find an open neighbourhood in \\(x\\) with that entire open neighbourhood maps to a subset of the open neighbourhood in \\(v\\).

<img src="/assets/manifold/Untitled 1.png" alt="220" style="zoom: 33%;" />

**Examples:**

The euclidian line, 34:00 in [https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4](https://www.youtube.com/watch?v=L1MC5GvlxPI&list=PLRlVmXqzHjUQHEx63ZFxV-0Ortgf-rpJo&index=4)

When you are dealing with manifolds, you are dealing with continuous functions from one manifold to another. The topologies in the manifolds are not the weird topologies shown in the video, they are all much more like the Euclidian topology. They are robust to capture everything we need, they are continuous and we are not dealing with mapping from one weird topology to another.

### Homeomorphisms

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


## Diffeomorphisms

### The curve

We define the curve as a mapping of \\(\lambda\\) to the differential manifold \\(X\\). 

Let \\(\lambda\\) be any point on an open line, \\(\lambda \in \mathbb{R}\\). 

Let the function \\(f\\) map \\(\lambda\\) to X,  \\(f(\lambda) \in X\\) because \\(f(\lambda): \mathbb{R} \rightarrow X\\)

<img src="/assets/manifold/Untitled-6515580.png" alt="120" style="zoom: 33%;" />


This picture tells that to construct curves and use them  for calculus, we can map then onto the differential manifold and then map them to a euclidean range, in this case \\(\mathbb{R}^2\\) with the usual topology. Hence to optain the coordinates in the chart, \\((a,b) = \gamma \circ f(\lambda_i) \in \mathbb{R}^2\\), for a given value of \\(\lambda\\). We really need the chart \\(\gamma\\) for this whole thing to work, otherwise we would not be able to obtain coordinates. 

Remember a chart is \\((U_i, \gamma_i)\\) (an open set / neighborhood around a point i and a continuous function mapping to the euclidean space.)

### Coordinate functions

If we have a chart \\((U, \gamma)\in \mathcal{A}\\), and a topological manifold: \\((X, \mathcal{T}_X, \mathcal{A})\\). 

We want to map a point \\(p\\) in the open cover / chart onto the euclidean space that has a usual topology. 

We know this can be done using \\(\gamma(p) = (\alpha^1, \alpha^2) =(a,b)\\), given that \\(\mathbb{R}^2\\).

We could also view \\(\gamma(p) = \left(\alpha^1(p), \alpha^2(p)\right)\\), here the \\(\alpha\\)s are coordinate functions. Each coordinate function takes a point \\(p \in U\\), \\(\alpha^i(p): U \Rightarrow \mathbb{R}\\). 

\\(\gamma\\) is then the functions that collects these coordinate functions together and construct a higher dimensional euclidian space.

### General relativity reference

<img src="/assets/manifold/Untitled 1-6515584.png" alt="12" style="zoom: 33%;" />

This picture shows how a new coordinate \\(X^{0^{\prime} }\\) is a function of old coordinates. 

\\(X^{0^{\prime} } (X^0, X^1)\\), The \\((X^0, X^1)\\) is the coordinates in the plane \\(\mathbb{R}^2\\) which is equivalent to \\(\gamma(p)\\). The \\(X^{0^{\prime} }()\\) then is a coordinate function that returns a real number which is just the \\(X^{0^{\prime} }\\) coordinate. In this sense, \\(X^{0^{\prime} } () = X^{0^{\prime} }\circ \gamma^{-1}\\). Combining them, we have the familiar expression: \\(X^{0^{\prime} } \circ \gamma^{-1} \left(\gamma(p)\right)\\)

The function \\(X^{0^{\prime} }\\) and the coordinate \\(X^{0^{\prime} }\\) is used redundantly in GR theory.

Of course, we can do the same for the coordinate \\(X^{1^{\prime} }\\).

Similarly, the old coordinates can be expressed as a function of the new coordinates:

\\(X^0 = X^0 \circ \phi^{-1}\left(\phi(p)\right)\\).

## Functions from one manifold to another

Is \\(f\\) a differentiable function?

\\(f : X \rightarrow Y\\), where \\(X\\) is a differentiable manifold \\((X, \mathcal{T}_X, \mathcal{A})\\), and \\(Y\\) is a differentiable manifold \\((Y,\mathcal{T}_Y,\mathcal{B})\\)


<img src="/assets/manifold/Untitled 2-6515589.png" alt="12123123" style="zoom: 33%;" />

Instead of using transition functions from one chart to another of an atlas, can we transition from one manifold to another? (this is marked with the purple line)

IF its differentiable everywhere, not just in the intersections, \\(f\\) is called a smooth differentiable function between the two manifolds. 

And as such, once again in order to understand the differentiability, we have to lean on all the structures of the manifolds. We need to lean on the charts, the chart maps \\(\gamma\\), the regions of the chart maps \\(U\\)

Just because \\(f\\) is differentiable, it does not necessarily mean that it is one-to-one and onto, and the inverse of \\(f\\) is also differentiable in the same way as \\(f\\) is. 


💡 However, IF this is the case that \\(f\\) is all of these things (basically everything in a homeomorphism and differentiability) then the two spaces are `diffeomorphic`.

> “Diffeomorphisms is the homeomorphisms for differentiable manifolds”

## Differentiable manifolds

TS → TM → DM

All differentiable manifolds are topological manifolds.

Not all topological manifolds are differentiable manifolds.

All topological manifolds are topological spaces.

Not all topologial spaces are topological manifolds.

What is the distinction between DM and TM?

**The topological manifold**: \\((X, \mathcal{T}_X, \mathcal{A})\\), where the atlas contains charts \\(\mathcal{A} = \{(U_i, \gamma_i)\}\\)

All of these charts cover the entire topological space \\(X\\), which forms the atlas. Locally each chart cover an open neighborhood which is an open subset of the topology. And each chart region is locally homeomorphic to some region of the euclidean space \\(\mathbb{R}^d = \mathbb{R} \times \mathbb{R} \times \dots \times \mathbb{R}_d\\) where \\(\times\\) is the cartesian product, which yields ordered pairs that can be used as coordinates in the euclidean space.

<img src="/assets/manifold/Untitled-6515497.png" alt="44444444" style="zoom: 33%;" />

We can do calculus in the euclidean space!

If we take a single point \\(p\\), which \\(U_1\\) centers around, it is essentially a large d-dimensional tuple, which is the coordinates of \\(p\\) in the euclidean space: \\(\gamma_1(p)=\{\alpha^1,\alpha^2,\dots, \alpha^d \}\\). The coordinates only makes sense in that specific chart \\((U_1, \gamma_1)\\). \\((U_2, \gamma_2)\\) has its own set of coordinates and are NOT comparable to other charts, because the charts are different euclidian spaces of the same dimensionality.

**What criteria must be made to make these topological manifolds differentiable?**

We need to understand transition functions, those transition functions are going to be maps to the different charts (when appropriate!).

### Transition functions

If we have an atlas \\(\mathcal{A} = \{ (U_1, \gamma_1), (U_2, \gamma_2), \dots \}\\) then we can look at the two open sets as \\(U_{1,2} = U_1 \cap U_2\\). \\(U_{1,2}\\) is a [simply connected](https://mathworld.wolfram.com/SimplyConnected.html) space and \\(\gamma\\)’s are homeomorphisms: 1-1, onto, continuous, the inverse is continuous. This means, they will preserve topological properties which means that the images keep these properties.

The sphere example from earlier showed a stereographic function, that overlapped almost entirely between the two sets aside from two points, the north and the south pole. 

Charts have overlaps, however, we do not need extreme overlaps.

Any point \\(p \in U_{1,2}\\), can be mapped into the regions of the euclidean space by either \\(\gamma_1(p)\\) or \\(\gamma_2(p)\\)

All the points in \\(U_{1,2}\\) will thus be mapped to some sub-region of \\(\gamma_1(U_1)\\) and \\(\gamma_2(U_2)\\).

We want to make a map from one euclidean sub-region of one chart to another euclidean subregion of another chart (\\(\mathbb{R}^d \rightarrow \mathbb{R}^d\\)). How do we build this map?

Let's say we want to map the point \\(p\\) from the euclidean range, obtained via \\(\gamma_1(p)\\). We simply need to leverage the fact that \\(\gamma\\) is bi-continuous, so we can easily map it back to the domain. Then, once we have \\(p\\) in the domain, we can map it to the euclidean range via \\(\gamma_2(p)\\):

$$
\gamma_2\left(\gamma_1^{-1}\left(\gamma_1(p)\right)\right)
$$

This looks like the standard functional notation. We don’t like this notation.

Instead we say \\(\gamma_2 \circ \gamma_1^{-1}(\gamma_1(p))\\), where \\(\gamma_1(p)\\) is the coordinates of p in the euclidean range, and \\(\gamma_2 \circ \gamma_1^{-1}\\) is simply a composed function that eats the coordinates of the point in the euclidean range.

**This is the transition function!**

It transitions from one chart to another! and we can take derivatives of this, because it maps from \\(\mathbb{R}^d \rightarrow \mathbb{R}^d\\)


<img src="/assets/manifold/Untitled 1-6515493.png" alt="r2" style="zoom: 33%;" />

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

### Example of transition functions: Coordinates

We go back to the sphere and we want to map it to \\(\mathbb{R}^2\\) with the usual topology.

<img src="/assets/manifold/Untitled 2-6515488.png" alt="r3" style="zoom: 33%;" />

In principle, the R2 is different from \\(\gamma_1(p)\\) and  \\(\gamma_2(p)\\), but we use it for illustrational ease. 

So it is clear that the composite function maps directly from the coordinate plane R2 to another coordinate plane R2.


## Recap 1: Constructing manifolds?

### Quick on topological manifolds

So a manifold is a topological space, which can be charted. These charts are part of an Atlas. 

\\((X, \mathcal{T}_X, \mathcal{A})\\). The charts map points or regions from the manifold to a euclidean space with the usual topology as explained in: 

- It is possible because a manifold is a topological space that is locally Euclidean

We remember the atlas was composed of charts \\((U, \gamma)\\), where U is an open set of the manifold and \\(\gamma\\) is

the mapping from the manifold to the euclidean space. The \\(\gamma\\)’s must be homeomorphic.

### Quick on differentiable manifolds
For a manifold to be differentiable, for any open set \\(U\\), we must be able to find a smooth transition function. \\(\gamma_i \circ \gamma_j^{-1} (\gamma_j(U_j))\\)

We know from the fact \\(\gamma\\)’s are homeomorphic that they are continuous, therefore, the composition of the two functions \\(\gamma\\) are continuous, \\(\gamma_i \circ \gamma_j^{-1}\\) is also continuous.

So let us just call \\(\gamma_i \circ \gamma_j^{-1} = \Psi\\) for now. We know we can differentiate on continuous functions, yet there are different degrees of differentiability as explained in  

It goes from \\(C^0\\) which is not differentiable but continuous to \\(C^{\infty}\\) which is infinitely differentiable.

Differentiable manifolds are therefore \\(C^1\\) and beyond. If nothing is stated, we assume that the transition functions are \\(C^{\infty}\\).

### Manifold learning?

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
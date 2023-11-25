---
layout: post
comments: true
title:  "Compactness, Connectedness and Topological properties"
excerpt: "An insightful exploration of Compactness, Connectedness and Topological properties"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Compactness, Connectedness and Topological properties

![Untitled](/assets/manifold/assets/Untitled-6515326.png){: .zoom50% }

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

## Compactness

Definition: A subset \\(S\\) of a topological space \\(X\\) is compact if for every open cover, there exists a finite subcover of \\(S\\).

Is the topological space \\((X, J_X)\\) compact?

First, let us ask about a subset. let the open set \\(A \subset (X, J_X)\\), is \\(A\\) compact?

### Covers, closed and open sets

An open cover of a subset \\(S\\) of \\(X\\) is a collection \\(\mathcal{C} = \{ U_\alpha \}\\) of open sets in \\(X\\) such that 

\\(\\)
S \subseteq \bigcup_\alpha U_{\alpha}
\\(\\)

A subcover is a sub collection \\(\mathcal{C}^{\prime} \subseteq \mathcal{C}\\) that still covers \\(S\\).

A subset \\(S\\) of \\(X\\) is compact if every open cover \\(\mathcal{C}\\) of \\(S\\) has a finite subcover.

An open set in a topological space \\(X\\) is a set \\(U\\) such that for every point \\(x \in U\\) there exists an open neighbourhood \\(N\\) of \\(x\\) such that \\(N\subseteq U\\). In simpler terms, this means that you can wiggle around any point in the set without ever stepping out of it.

A closed set in a topological space \\(X\\) is a set \\(C\\) such that its complement \\(X \setminus C\\) is open. This means that a closed set contains all its `boundary points`. A boundary point of \\(S\\) is a point \\(x\\) such that every open neighborhood of \\(x\\) intersects both \\(S\\) and \\(X \setminus S\\). 

1. **Open Sets and Compactness**: Open sets don't contain their boundary points. When you try to cover an open set with smaller open sets (an open cover), you'll find that you can't pin down the boundary with a finite number of them. You'll always need "just one more" to get closer to the boundary, leading to an infinite subcover.
2. **Closed Sets and Compactness**: Closed sets do contain their boundary points. When you cover a closed (and bounded) set with open sets, you can always find a finite subcover that includes the boundary. This makes the set compact, at least in Euclidean spaces.

### Back to the example

The number of open sets we need to cover \\(A\\) is 1. Obviously we can cover \\(A\\) with \\(1, \dots, \infty\\)

We can use infinite open sets in two ways:

- Make a cover with infinite amount of open sets to cover the subset. (an infinite cover with no finite sub-cover)
- Make an cover with infinite amount of open sets that cover the whole space, thereby ultimately cover the subset with some finite number of these open sets. (an infinite cover with finite sub-cover)

Example: 

The video uses an example of a rectangle in the usual topology of the plane. It compares an open rectangle in which you can fill up the interior of the rectangle (which is equal to the open rectangle itself, by an infinite amount of open subsets. This is then the cover. For this cover, therefore, there is no finite subset, since it would then not be a cover.

Contrary, the closed rectangle. By a similar fashion we can begin to cover the rectangle by an infinite amount of open sets. Here we get infinitely close to the border of the rectangle. However, by this token, we cannot reach the border and thus we need an additional cover to cover the rectangle. Because of this, there will exist a finite sub-cover for the open cover. 

## Connectedness

Definition: A connected set is a set that cannot be partitioned into two non-empty subsets which are open in the relative topology induced on the set. Equivalently, it is a set which cannot be partitioned into two non-empty subsets such that each subset has no points in common with the set closure of the other.

If we have a topological space \\((X, J_X)\\) 

![Untitled](/assets/manifold/assets/Untitled 1-6515321.png){: .zoom50% }

Where the space X consists of two spaces G and H, let G and H be open sets in the topology, if there is no intersection between them, then we can say that the space \\(X\\) is NOT connected.

By the property of homeomorphism, if we map to another topological space, the property of connectedness follows:

![Untitled](/assets/manifold/assets/Untitled 2.png){: .zoom50% }

**Path-connectedness**

[https://mathworld.wolfram.com/Pathwise-Connected.html](https://mathworld.wolfram.com/Pathwise-Connected.html)

Path-connectedness is another way of understanding connectedness

If we have a topological space \\(X \in \mathbb{R}^2\\)

and we have a continuous function \\(f\\) from \\([0,1]\\) to \\(X\\) such that \\(f(0) = x\\) and \\(f(1)=y\\). 

If you are path connected, you are connected.

![Untitled](/assets/manifold/assets/Untitled 3.png){: .zoom50% }

## Homotopy

Definition: A continuous transformation from one function to another. A homotopy between functions \\(f\\) and \\(g\\) from a space \\(X\\) to a space \\(Y\\) is a continuous map \\(G\\) from \\(X \times [0,1] \rightarrow Y\\) such that \\(G(x,0)=f(x)\\) and \\(G(x,1)=g(x)\\) where \\(\times\\) denotes set pairing. another way of saying this is that a homotopy is a path in the mapping space \\(\operatorname{Map}(X,Y)\\) from the first function to the second. Two mathematical objects are said to be [homotopic](https://mathworld.wolfram.com/Homotopic.html) if one can be continuously deformed into the other.

![Untitled](/assets/manifold/assets/Untitled 4.png){: .zoom50% }

The doughnut have different homotopies of paths because of the hole in the space. I.e. the blue function of path-connectedness behaves vastly different from that of the red line, and the green line and vice versa. However in the topological space of \\(X\\), the red line and the blue line and the green line can be deformed to the other (this is called “simply connected” \\(C \sim X\\)).

The properties:

- separability
- countability
- compactness
- connectedness

Are all part of the topological space. That are carried over via homeomorphism, that allow us to deform.

We do not have notions of shape, distance and other measurements.
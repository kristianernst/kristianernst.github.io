---
layout: post
comments: true
title:  "Separability"
excerpt: "separability in topology"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Separability

## The Base of the topology

We use a Line as examples:

Different options:

- the Euclidean topology, simply the line
	- the usual line (open ball topology)
- The nested interval: (0, 1 - 1/n)
	- nested line (same set but with a different topology!)
- closed interval topology: X is the set (-1, 1)
	- base constituted of half open intervals: [-1, a) (b,1] where a > 0 and b < 0.

## Separability

\\(T_0\\) separability:

For any two points \\(x, y\\) there is an open set \\(U\\) such that \\(x \in U\\) and \\(y \notin U\\) or \\(y \in U\\) and \\(x \notin U\\). A space fulfilling this axiom is called a T0-space.

It is easy in the euclidean topology, where you just take an interval that only includes one point.

\\(T_1\\) separability; frechet 

For any two points \\(x, y \in X\\) there exists two open sets \\(U\\) and \\(V\\) such that \\(x \in U, y \notin U\\) and \\(y \in V, x \notin V\\). A space satisfying this axiom is known as a [T1-space](https://mathworld.wolfram.com/T1-Space.html)

\\(T_2\\) separability; Hausdorff (This is the one you usually encounter!)

For any two points \\(x, y \in X\\) there exists two open sets \\(U, V\\) such that \\(x \in U, y \notin U\\) and \\(y\in V, x \notin V\\), and \\(U \cap V = \emptyset\\)

This is an extension of T1 that also says that the two sets V and U has to be disjoint!

![Untitled](/assets/manifold/assets/Untitled-6515138.png){: .zoom50% }

It is clear that if \\(T_1\\) is true, then \\(T_0\\) is automatically true, however not the other way around.

Here is the complete separability axiom ranking:

\\(T_4 \Rightarrow T_3  \Rightarrow T_2  \Rightarrow T_1  \Rightarrow T_0\\)
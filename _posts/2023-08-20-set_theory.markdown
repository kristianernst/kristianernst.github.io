---
layout: post
comments: true
title:  "Countability"
excerpt: "Exploring the concepts of countable and uncountable sets in set theory."
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Countability

In set theory, a set is said to be countable if its elements can be put into a one-to-one correspondence with the set of natural numbers N={0,1,2,3,…} or a subset of N. In simpler terms, if you can "count" the elements of a set using natural numbers, then the set is countable.

Two types of countable sets:

- countably infinite sets: these sets have infinitely many elements but you can still establish a one-to-one correspondence with the natural numbers
- Finite sets: these sets have finitely many elements and are trivially countable

**Rational numbers: countable**

Rational numbers are numbers that can be expressed as a fraction: \\(\frac{p}{q}\\) where \\(p\\) and \\(q\\) are integers and \\(q \neq 0\\). The set of rational numbers is denoted \\(\mathbb{Q}\\).

*Proof of countability of rational numbers:*

We can arrange rational numbers in a grid where the numerator \\(p\\) is on one axis and the denominator \\(q\\) is on another axis. We start from \\(1/1\\) and move in a zigzag manner to cover all possible fractions. This way, each rational number can be mapped to a unique natural number.

![image-20231005153539785](assets/set/image-20231005153539785.png){: .zoom-50}

*Irrational numbers: uncountable*

The set of irrational numbers \\(\mathbb{I}\\) cannot be expressed as a simple fraction. 

This set is uncountable as proven from [Cantor’s diagonal argument](https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument).

To show this, we use a proof by contradiction in which we assume that the set of irrational numbers \\(\mathbb{I}\\) can be mapped to the natural numbers \\(\mathbb{Z}^+\\).  

We take a subset \\((0,1) \Leftrightarrow \mathbb{Z}^+\\)

```
0. d11 d12 d13 d14 ...
0. d21 d22 d23 d24 ...
0. d31 d32 d33 d34 ... 
.   .   .   .   .  .
.   .   .   .   .   .
.   .   .   .   .    .
```

Cantor's diagonal argument says, "Let's create a new number that differs from the n’th number in the n’th decimal place." 

Lets call this number \\(a = 0.a_1a_2a_3\cdots\\), where \\(a_i \neq d_{ii}\\)

The new number we've created is clearly between 0 and 1, and it's not on our list because it differs from every number in the list. This contradicts our assumption that we could list all the irrational numbers, proving that the set \\(\mathbb{I}\\) is uncountable. Even if we just added this new number to the list, we would be able to create a new number thus making it uncountable again.


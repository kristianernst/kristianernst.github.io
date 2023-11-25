---
layout: post
comments: true
title:  "Linear algebra"
excerpt: "Linear algebra"
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---


# Linear algebra

I took a lot of inspiration from the books: [mathematics for machine learning](https://mml-book.github.io/) and [deep learning](https://www.deeplearningbook.org/) and Kahn Academy when writing these notes. 

# Table of contents
- [Linear algebra](#linear-algebra)
- [Table of contents](#table-of-contents)
- [Notation of linear algebra](#notation-of-linear-algebra)
  - [Vectors](#vectors)
  - [Matrix](#matrix)
  - [The identity matrix and diagonal matrices](#the-identity-matrix-and-diagonal-matrices)
- [Norms](#norms)
- [The Manhattan norm (\\(L\_1\\)):](#the-manhattan-norm-l_1)
- [The Euclidian norm (\\(L\_2\\)):](#the-euclidian-norm-l_2)
- [Generalization (what is a norm?)](#generalization-what-is-a-norm)
  - [Norms for matrices](#norms-for-matrices)
- [Norms and their use in regularization](#norms-and-their-use-in-regularization)
  - [What is regularization?](#what-is-regularization)
  - [Norms in regularization](#norms-in-regularization)
- [The determinant](#the-determinant)
  - [**Properties of the determinant**](#properties-of-the-determinant)
  - [The classical adjoint of a matrix](#the-classical-adjoint-of-a-matrix)
- [Linear independence and rank](#linear-independence-and-rank)
- [Inverse of a square matrix](#inverse-of-a-square-matrix)
- [Matrix multiplication properties](#matrix-multiplication-properties)
- [Matrix multiplication is](#matrix-multiplication-is)
- [The identity matrix](#the-identity-matrix)
- [Diagonal matrix:](#diagonal-matrix)
- [Symmetric and antisymmetric matrices](#symmetric-and-antisymmetric-matrices)
- [Trace](#trace)



# Notation of linear algebra

## Vectors

\\(\boldsymbol{v} \in \mathbb{R}^d\\), the vector v is an element in a set. The set is a d-dimensional real set.

if \\(\mathbb{R}^{d=2}\\), we can ascribe the vector to a 2 dimensional space (imagine a coordinate system with x and y axis)

Column vector:

\\(\boldsymbol{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_d\end{bmatrix}\\)

Row vector:

\\(\boldsymbol{v}^T = [v_1, v_2, \dots v_d]\\)

## Matrix

\\(\boldsymbol{A} \in \mathbb{R}^{m\times n}\\), is a matrix which is an element of the real set of m*n dimensionality.

**Matrix interpretation**

\\(\boldsymbol{A} = \begin{bmatrix} A_{11}, & A_{12}, & \dots, & A_{1,n} \\ A_{21}, & A_{22}, & \dots, & A_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{m1}, & A_{m2}, & \dots, & A_{mn} \end{bmatrix}\\)

Matrix as column vectors

$$
\begin{align}
\boldsymbol{A} = \begin{bmatrix} \boldsymbol{c}_{1}, \boldsymbol{c}_{2}, \dots,\boldsymbol{c}_{n} \end{bmatrix}
\end{align}
$$

Where:

$$
\begin{align}
\boldsymbol{c}_1 = \begin{bmatrix} \boldsymbol{A}_{11} \\ \boldsymbol{A}_{21} \\ \vdots \\ \boldsymbol{A}_{m1} \end{bmatrix}
\end{align}
$$

**Matrix as row vectors**

$$
\begin{align}
\boldsymbol{A} = \begin{bmatrix} \boldsymbol{r}_{1}\\ \boldsymbol{r}_{2}\\ \vdots \\ \boldsymbol{r}_{m} \end{bmatrix}
\end{align}
$$


Where:

$$
\begin{align}
\boldsymbol{r}_1 = \begin{bmatrix} \boldsymbol{A}_{11}, \boldsymbol{A}_{12}, \dots, \boldsymbol{A}_{1n} \end{bmatrix}
\end{align}
$$


**The transpose of a matrix**

Given the matrix
\\(\boldsymbol{A}=\left[\begin{array}{cccc} A_{11}, & A_{12}, & \ldots, & A_{1, n} \\ A_{21}, & A_{22}, & \ldots, & A_{2 n} \\ \vdots, & \vdots, & \vdots, & \vdots \\ A_{m 1}, & A_{m 2}, & \ldots, & A_{m n} \end{array}\right]\\)

The transpose of \\(\boldsymbol{A}\\) is given by:
\\(\boldsymbol{A}^T=\left[\begin{array}{cccc} A_{11}, & A_{21}, & \ldots, & A_{m 1} \\ A_{12}, & A_{22}, & \ldots, & A_{m 2} \\ \vdots, & \vdots, & \vdots, & \vdots \\ A_{1 n}, & A_{2 n}, & \ldots, & A_{m n} \end{array}\right]\\)

Hence, 

$$
\begin{align}
\boldsymbol{A}_{ij} = \boldsymbol{A}^T_{ji}
\end{align}
$$

, i.e., the rows and columns has been swapped. 

## The identity matrix and diagonal matrices

The identity matrix has 1 along its diagonal and 0 elsewhere

$$
\begin{align}
\boldsymbol{I} = \begin{bmatrix} 1, &0, &\dots,& 0 \\ 0,& 1,& \dots, & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0, &0, & \dots, &1 \end{bmatrix}
\end{align}
$$

Similarly, a diagonal matrix has non-zero values along the diagonal and 0 elsewhere.

**Symmetric matrices**

A symmetric matrix: \\(\boldsymbol{A} = \boldsymbol{A}^T\\) 

In that sense, we can say that for symmetric matrices, its row vectors is equal to its column vectors!

$$
\begin{align}
\boldsymbol{A}=\left[\begin{array}{lll} a & b & c \\ b & d & e \\ c & e & f \end{array}\right] = \boldsymbol{A}^T
\end{align}
$$

**The trace of a matrix**

The trace of a matrix is the sum of its diagonal: \\(\operatorname{tr}(\boldsymbol{A}) = \sum_i A_{ii}\\)

# Norms

Norms are used to measure the length of a vector in some way. There exists many types of norms. We will cover two types: \\(L_1\\) and \\(L_2\\) norms. 

# The Manhattan norm (\\(L_1\\)):

Formula:

$$
\| \boldsymbol{v}\| _1 = \sum_{i=1}^n \mid v_i\mid 
$$

The manhattan norm is simply obtained by summing the absolute values of each vector component. 

# The Euclidian norm (\\(L_2\\)):

Formula:

$$
\| \boldsymbol{v}\| _2 = \sqrt{\sum_{i=1}^n v_i^2}
$$

The euclidian norm is obtained by taking the square root of the summation of the squared vector components.

Note that \\(\boldsymbol{x}^T\boldsymbol{x} = \| \boldsymbol{x}\| _2^2\\)

# Generalization (what is a norm?)

A norm is any function \\(f: \mathbb{R}^n \rightarrow \mathbb{R}\\) that satisfies these 4 properties:

1. For all \\(x \in \mathbb{R}^n\\), \\(f(x)\geq 0\\) (non-negativity).
2. \\(f(x) = 0\\) iff \\(x=0\\) (definiteness).
3. \\(\forall x \in \mathbb{R}^n, t\in \mathbb{R}, f(tx) = \mid t\mid f(x)\\) (homogeneity).
4. \\(\forall x, y \in \mathbb{R}^n, f(x + y) \leq f(x) + f(y)\\) (triangle inequality).

The euclidian and manhattan norm can be expressed with the following formula:

$$
\| x\| _p = \left(\sum_{i=1}^n \mid x_i\mid ^p\right)^{\frac{1}{p} }
$$

<img src="/assets/linalg/norm.png" alt="norm" style="zoom:33%;" />

The image above provides a visual aid in understanding the functioning of the different types of norms. The lines in the x,y coordinate system indicates for what x,y combinations the resulting norm is equal to 1.

- For the L1 norm it is simply the absolute value of the x component + the absolute value of the y component. hence \\(\mid -.5\mid  + \mid .5\mid  = 1\\)  is the middle point of the line in the bottom right quadrant of the coordinate system.
	- In this light, we can view the norm as the distance travelled by moving along the different dimensions of the system. In this case we move 0.5 along the x axis and 0.5 along the y axis.
- For the L2 norm, we get a circular form. hence when x and y are apprx. = .706 we get L2 = 1.
	- In this light, we view the norm as the distance travelled from origo to a point  vector constructed by the component trav

## Norms for matrices

An example of a matrix norm is the *Frobenius* norm:

$$
\| \boldsymbol{A}\| _F = \sqrt{\sum_{i=1}^m\sum_{j = 1}^n A_{ij}^2} =\sqrt{\operatorname{tr}(\boldsymbol{A}^T\boldsymbol{A})}  \ .
$$

# Norms and their use in regularization

Both the L1 and L2 norms are used frequently for regularization in machine learning. 

There are many great sources that explains use cases for both norms. Therefore, I provide links to these along the way.

## What is regularization?

Ian Goodfellow et. al. coins the use of regularization as: 

> “Any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error” - [book](https://www.deeplearningbook.org/)

In effect, what we often come to care about is reducing overfitting of the model, often resulting in increasing training error but reducing prediction error on future data.

Many types of regularization techniques exists. Generally, we can group regularization into two broad categories:

1. **Explicit Regularization**: This involves explicitly adding a term to the optimization problem. These terms could be priors, penalties, or constraints. The regularization term, or penalty, imposes a cost on the optimization function to make the optimal solution unique.
2. **Implicit Regularization**: This includes techniques like early stopping, using a robust loss function, and discarding outliers. For instance, in deep learning, training a model using stochastic gradient descent with early stopping is a form of implicit regularization.

source: [wikipedia](https://en.wikipedia.org/wiki/Regularization_(mathematics))

## Norms in regularization

The usage of norms in regularization falls under the category *explicit* regularization.

Here, norms are used as penalisation terms. 

Stealing notations from [book](https://www.deeplearningbook.org/),  we can express a regularized objective function \\(\tilde{J}\\) as follows:

$$
\tilde{J}(\boldsymbol{\theta}; \boldsymbol{X}, \boldsymbol{y}) = J(\boldsymbol{\theta}; \boldsymbol{X}, \boldsymbol{y}) + \alpha \Omega(\boldsymbol{\theta})
$$

Here, \\(\alpha\\) is a hyperparameter that is often determined by cross validation. Its function is to weigh the contribution of the norm penalty term \\(\Omega\\), relative to the standard objective function. Therefore, by setting \\(\alpha = 0\\), \\(\tilde{J} = J\\).

**L1 norm as regularization term (Lasso regularization)**

This adds the sum of the absolute values of the coefficients as penalty term to the loss function. It is pretty “rough” leading to some feature coefficients becoming exactly zero which is equivalent to completely exploding those features from the model.

**L2 norm as regularization term (Ridge regularization / weight decay)**

This term adds the sum og squared values of coefficients as a penalty term to the loss function. The result of adding this penalty is shrinked coefficients for less important features. In other words, we drive all weights closer to the origin (origin = zero when we do not know / have any qualified guesses of the true point in space).

In contrast to using L1 norm as regularization term, the L2 norm keeps a dense model where all features are given some importance.


# The determinant

The determinant of a square matrix is a function that returns a scalar.

\\(\boldsymbol{A} \in \mathbb{R}^{n \times n}\\), \\(\operatorname{det} : \mathbb{R}^{n\times n} \rightarrow \mathbb{R}\\)

Consider the set of points \\(S \subset \mathbb{R}^n\\) formed by taking all possible linear combinations of the row vectors \\(\boldsymbol{a}_1^T, \boldsymbol{a}_2^T, \dots, \boldsymbol{a}_n^T \in \mathbb{R}^n\\) of \\(\boldsymbol{A}\\), where the coefficients of all linear combinations are between \\(0\\) and \\(1\\).

Formally put:

$$
S = \{\boldsymbol{v}\in\mathbb{R}^n:\boldsymbol{v} = \sum_{i=1}^n \alpha_i \boldsymbol{a}_i \quad \text{where } 0 \leq \alpha_i \leq 1, i = 1, 2, \dots, n\} \ .
$$

Let \\(\boldsymbol{A} = \begin{bmatrix} 1 & 3 \\ 3 & 2 \end{bmatrix}\\), here \\(\boldsymbol{a}_1 = \begin{bmatrix} 1 \\ 3\end{bmatrix}\\) and \\(\boldsymbol{a}_2 = \begin{bmatrix} 3 \\ 2 \end{bmatrix}\\).

In this case, \\(S\\) has the shape of a parallelogram. 

The value of the determinant  \\(\mid \boldsymbol{A}\mid  = -7\\).

The area of the paralellogram is also \\(7\\). 

<img src="/assets/linalg/image-20231005155711707.png" alt="image-20231005155711707" style="zoom:50%;" />

In three dimensions, the determinant of a matrix is usually a parallelepiped (box with parallelogram-shaped surfaces). 

In this light, the determinant tells us something about the “volume” of the set \\(S\\). 

## **Properties of the determinant**

Algebraically, the determinant follows these three properties (from which all other properties follow)

1. The determinant of the identity matrix is 1. \\(\mid \boldsymbol{I}\mid =1\\).

2. Given a matrix \\(\boldsymbol{A}^{n\times n}\\), if we multiply a single row in \\(\boldsymbol{A}\\) by a scalar \\(t\\), then the determinant of the new matrix is \\(t\mid \boldsymbol{A}\mid \\) (geometrically, multiplying one of the sides of the set \\(S\\) by a factor of \\(t\\) causes the volume to increase by \\(t\\)

    $$
    \begin{align}
    \left\lvert \begin{bmatrix} - & t a_1^T & - \\ - & a_2^T & - \\ & \vdots & \\ - & a_m^T & - \end{bmatrix} \right\rvert = t \lvert A \rvert .
    \end{align}
    $$


3. If we exchange any two rows, the determinant of the new matrix is \\(-\mid \boldsymbol{A}\mid \\).

    $$
    \begin{align}
    \left\lvert \begin{bmatrix} - & a_2^T & - \\ - & a_1^T & - \\ & \vdots & \\ - & a_m^T & - \end{bmatrix} \right\rvert = -\lvert A \rvert .
    \end{align}
    $$


Given these three properties, here’s other useful ones:

- For \\(\boldsymbol{A} \in \mathbb{R}^{n\times n}, \mid \boldsymbol{A}\mid  = \boldsymbol{A^T}\\)
- For \\(\boldsymbol{A}, \boldsymbol{B} \in \mathbb{R}^{n\times n}, \mid \boldsymbol{A}\boldsymbol{B}\mid  = \mid \boldsymbol{A}\mid  \mid \boldsymbol{B}\mid .\\)
- For \\(\boldsymbol{A} \in \mathbb{R}^{n\times n}, \mid \boldsymbol{A}\mid  = 0\\) iff \\(\boldsymbol{A}\\) is singular (If A is singular it does not have full rank and hence columns are linearly dependent. In this case, \\(S\\) corresponds to a “flat sheet” within the n-dimensional space and hence has zero volume).
- For \\(\boldsymbol{A} \in \mathbb{R}^{n\times n}\\) and \\(\boldsymbol{A}\\) is non-singular: \\(\mid \boldsymbol{A}^{-1}\mid  = 1/\mid \boldsymbol{A}\mid \\)

 

Before giving the general definition for the determinant, we define, for \\(\boldsymbol{A} \in \mathbb{R}^{n \times n}, \boldsymbol{A}_{\backslash i, \backslash j} \in \mathbb{R}^{(n-1) \times(n-1)}\\) to be the matrix that results from deleting the \\(i\\)th row and \\(j\\)th column from \\(\boldsymbol{A}\\). 

The general (recursive) formula for the determinant is

$$
\begin{align}
\lvert \boldsymbol{A}\rvert  &= \sum_{i=1}^n (-1)^{i+j} a_{ij} \lvert \boldsymbol{A}_{\backslash i, \backslash j}\rvert \quad (\text{for any } j \in 1, \ldots, n) \\
&= \sum_{i=1}^n (-1)^{i+j} a_{ij} \lvert \boldsymbol{A}_{\backslash i, \backslash j}\rvert \quad (\text{for any } i \in 1, \ldots, n)
\end{align}
$$


Example:

<img src="/assets/linalg/Untitled 1.png" alt="Untitled" style="zoom:50%;" />

## The classical adjoint of a matrix

The adjoint of a matrix \\(\boldsymbol{A} \in \mathbb{R}^{n \times n}\\) is denoted: \\(\operatorname{adj}(\boldsymbol{A})\\)

$$
\operatorname{adj}(\boldsymbol{A})\in \mathbb{R}^{n\times n}, \quad \operatorname{adj}(\boldsymbol{A})_{ij} = (-1)^{i+j}\mid \boldsymbol{A}_{\setminus j , \setminus i}\mid .
$$

It can be shown that for any non-singular \\(\boldsymbol{A} \in \mathbb{R}^{n \times n}\\), \\(\boldsymbol{A}^{-1} = \frac{1}{\mid \boldsymbol{A}\mid }\operatorname{adj}(\boldsymbol{A})\\).

*Proof:*

1. \\(\boldsymbol{A} \cdot  \operatorname{adj}(\boldsymbol{A}) = \mid \boldsymbol{A}\mid  \cdot \boldsymbol{I}\\)  (the matrix A multiplied with its adjugate, results in a diagonal matrix with values of the determinant of A)

2. Multiply both sides by \\(\frac{1}{\boldsymbol{A} }\\):  

\\(\frac{1}{\boldsymbol{\mid A\mid } } \cdot \boldsymbol{A} \cdot \operatorname{adj}(\boldsymbol{A}) = \frac{1}{\boldsymbol{\mid A\mid } }\mid \boldsymbol{A}\mid  \cdot \boldsymbol{I} = \boldsymbol{I}\\)

3. Rearrange to find the inverse:

\\(\frac{1}{\boldsymbol{\mid A\mid } } \cdot \operatorname{adj}(\boldsymbol{A}) = \boldsymbol{I} \cdot \frac{\boldsymbol{1} }{\boldsymbol{A} } = \boldsymbol{A}^{-1}\\)

I want to prove that \\(Z = A^T \operatorname{adj}(A) = \mid A\mid ^TI\\)

Step 1):

$$
\begin{align}
Z_{ij} = \sum_{k=1}^n A_{ki} (\operatorname{adj}(A))_{kj} = \sum_{k=1}^n A_{ki} \left((-1)^{j+k} \mid A_{\setminus j \setminus k}\mid \right) = \sum_{k=1}^n (-1)^{j+k} A_{ki} \mid A_{\setminus j \setminus k}\mid
\end{align}
$$

step 2) We observe that for \\(i = j\\) we actually have the formula for the determinant using the Laplace Expansion formula:

\\(\mid A\mid  = \sum_{k=1}^n (-1)^{i+j} A_{ij} \mid A_{\setminus i \setminus j}\mid \\) for either: all \\(i\\) and one \\(j\\) OR all \\(j\\) and one \\(i\\).

For \\(i \neq j\\), we replace the ith column of a with the kth one. This results in a matrix with two identical columns and so of rank < n. Therefore, the determinant is 0.

It has therefore been shown that all diagonal elements of \\(Z\\) is equal to the determinant of \\(A\\) and non-diagonal elements \\(=0\\).


# Linear independence and rank

A set of vectors \\(\{\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_n\} \subset \mathbb{R}^m\\) is said to be linearly independent if no vector can be represented as a linear combination of the remaining vectors. Conversely, if one vector is linearly dependent, it can be construed by a linear combination of some of the remaining vectors.

Linear dependence of vector x_n expressed as a formula:

$$
\boldsymbol{x}_n = \sum_{i=1}^{n-1} \boldsymbol{\alpha}_i \boldsymbol{x}_i
$$

where \\(\boldsymbol{\alpha}_i \in \mathbb{R}\\) is some scalar value.

When the opposite is true:

$$
\boldsymbol{x}_n \neq \sum_{i=1}^{n-1} \boldsymbol{\alpha}_i \boldsymbol{x}_i
$$

vector x_n *is* linearly dependent.

**Rank**

The `column rank` of a matrix \\(A \in \mathbb{R}^{m\times n}\\) is the size of the largest subset of columns of \\(A\\) that constitute a linearly independent set.

The `row rank` of the same matrix is the size of the largest subset of rows of \\(A\\) that constitute a linearly independent set.

Property: For any matrix, the row rank = column rank, we therefore refer to both collectively by the `rank` of a matrix.

General properties:

1. A matrix is set to have `full rank` when the following is true:

$$
A\in \mathbb{R}^{m\times n }, \quad \operatorname{rank}(A) =\operatorname{min}(m,n) \ .
$$

1. A matrix does not have full rank if the following is true: \\(\operatorname{rank}(A) < \operatorname{min}(m,n) \\\) . 
2. For \\(A\in \mathbb{R}^{m\times n }, \ B \in \mathbb{R}^{n\times p},  \ \operatorname{rank}(AB) \leq \operatorname{min}(\operatorname{rank}(A), \ \operatorname{rank}(B)) \ .\\)
3. \\(A, B\in \mathbb{R}^{m\times n }, \ \operatorname{rank}(A + B) \leq \operatorname{rank}(A) + \operatorname{rank}(B) \ .\\)

# Inverse of a square matrix

The inverse of a square matrix \\(A \in \mathbb{R}^{n \times n}\\) is denoted \\(A^{-1}\\) and is a unique matrix. 

Truths: 

- Only square matrices can have an inverse
- Not all square matrices have inverses
- \\(A\\) is `invertible` and `non-singular` if \\(A^{-1}\\) exists
- \\(A\\) is `non-invertible` and `singular` if \\(A^{-1}\\) does not exist.
- In order for \\(A\\) to have an inverse, it must be full rank.

If a square matrix has an inverse, it is the unique matrix such that:

$$
A^{-1}A=I=AA^{-1}
$$

Properties of non-singular matrices:

1. \\((A^{-1})^{-1}=A\\)
2. \\((AB)^{-1} = B^{-1}A^{-1}\\)
3. \\((A^{-1})^T = (A^T)^{-1}\\), (for this reason, this matrix is often denoted \\(A^{-T}\\).


# Matrix multiplication properties

# Matrix multiplication is

`Associative` : \\((AB)C = A(BC)\\)

`Distributive` : \\(A(B+C) = AB + AC\\)

In general, NOT `commutative` : it is not always true that \\(AB = AC\\).

# The identity matrix

\\(I \in \mathbb{R}^{n \times n}\\), where \\(I_{ij} = \begin{cases} 1 \quad i = j \\ 0 \quad i \neq j \end{cases}\\)

For all \\(A \in \mathbb{R}^{m \times n}\\): \\(AI = A = IA\\)

Note that in some sense, the notation for the identity matrix is ambiguous, since it does not specify the dimension of I.

# Diagonal matrix:

\\(D \in \mathbb{R}^{m\times n}\\), where \\(D_{ij} = \begin{cases} d_i & & i = j \\ 0 & & i \neq j \end{cases}\\)

\\(D = \operatorname{diag}(d_1, d_2, \dots, d_n)\\)

Therefore, \\(I = \operatorname{diag}(1, 1, \dots, 1)\\).

# Symmetric and antisymmetric matrices

\\(A \in \mathbb{R}^{m \times m}\\) is symmetric if \\(A^T = A\\). 

\\(A\\) is antisymmetric if \\(A = -A^T\\).

For any square matrix, the matrix \\(A + A^T\\) is symmetric and the matrix \\(A - A^T\\) is antisymmetric.

By these conditions, a matrix \\(A\\) is viewed as a composite of a symmetric and an anti-symmetric matrix: \\(A = \frac{1}{2} (A + A^T) + \frac{1}{2} (A - A^T)\\).

\\(A \in \mathbb{S}^n\\) means that \\(A\\) is a symmetric \\(n \times n\\) matrix.

# Trace

The trace of a square matrix is the sum of the diagonal:

\\(\operatorname{trace}(A) = \sum_{i = 1}^n A_{ii}\\)

**Can you take the trace of a non-square matrix?**

- No. The reason is that there is not a consistent diagonal that spans the entire matrix from the top-left to the bottom-right.

We can prove that the following properties hold:

1. For \\(A \in \mathbb{R}^{n\times n}\\),  \\(\operatorname{tr}A = \operatorname{tr}A^t\\)
2. For \\(A, B \in \mathbb{R}^{n\times n}, \operatorname{tr}(A+B) = \operatorname{tr} A + \operatorname{tr} B\\)
3. For \\(A \in \mathbb{R}^{n\times n}, t\ \in \mathbb{R}, \operatorname{tr}(t A) = t \operatorname{tr}A\\).
4. For \\(A, B\\) s.t. \\(AB\\) is square, \\(\operatorname{tr} AB = \operatorname{tr} BA\\).
5. For \\(A, B, C\\) s.t. \\(ABC\\) is square, \\(\operatorname{tr} ABC = \operatorname{tr} BAC = \operatorname{tr} CAB, \dots\\) and son for the product of more matrices.



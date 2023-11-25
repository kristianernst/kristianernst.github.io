---
layout: post
comments: true
title:  "Multivariate calculus"
excerpt: "The gradient is the key of multivariate calculus."
date:   2023-08-20 22:00:00
category: "Math"
mathjax: true
---

# Multivariate calculus


- [Multivariate calculus](#multivariate-calculus)
	- [Gradients](#gradients)
	- [Directional derivatives](#directional-derivatives)
		- [Proof that the gradient vector point in the direction of steepest ascent](#proof-that-the-gradient-vector-point-in-the-direction-of-steepest-ascent)


## Gradients

The gradient is the key of multivariate calculus.

If one knows how to take a partial derivative of a function, one knows how to compute the gradient. 

Example 1:

If we have the function:

$$
f(x,y) = sin(x)y^2
$$

Then we can compute the two partial derivatives:

$$
\begin{align*}
\frac{\partial f}{\partial x } = cos(x)y^2, & \quad & \frac{\partial f}{\partial y} = 2sin(x)y \ .
\end{align*}
$$

The gradient of $$f$$ is just a vector containing all the partial derivatives of its function:

$$
\nabla f = \begin{bmatrix}\frac{\partial f}{\partial x } = cos(x)y^2 \\ 
\frac{\partial f}{\partial y} = 2sin(x)y \end{bmatrix}
$$

Example 2: 

Say we have the function: $$f(x,y)=x^2+y^2$$ 

It looks like this:

<img src="/assets/multical/image-20231005154516751.png" alt="image-ri" style="zoom:50%;" />


We can plot the gradients at different positions in the xy-plane, the direction of the arrows indicate the direction of steepest ascent. The length of the arrow represents the magnitude.


<img src="/assets/multical/image-20231005154532413.png" alt="image-ra" style="zoom:50%;" />

- Knowledge check: how is one of these vectors derived?

	These vectors are computed by the gradient of the function:

	$$
	\nabla f = \begin{bmatrix} 2x \\ 2y
	
	\end{bmatrix}
	$$

	**Steepest Ascent**: The gradient vector at a point always points in the direction of the steepest ascent of the function at that point. This means that if you were to move in the direction of the gradient vector, you would be moving in the direction where the function increases the fastest.

- Knowledge check: what is the relationship between the vectors and contour lines?

	Contour lines are a topological mapping of a function. We could draw curves of $$f(x,y)= x^2 + y^2$$, where each line results in $$f = 1,2,3,4,5,...$$ etc.

	Now imagine the contour lines are drawn with very small intervals. We would expect that the contour lines were almost parallel to each other. Because of this property, if we were to ascend the function as fast as possible, we would do it by finding the shortest path between each contour line. 

	Due to parallelism between the two nearest contour lines, the shortest path is a direction orthogonal to the contour line we are at. 

	We know that the gradient vectors are the ones point in the direction of steepest ascent. Therefore, the gradient vectors are orthogonal to the contour lines

	
	<img src="/assets/multical/image-20231005154554850.png" alt="image-rl" style="zoom:50%;" />

## Directional derivatives

Directional derivatives are used to measure the rate a function changes as we transgress along a certain direction. 

$$
D_{\boldsymbol{w} } f(x) =   \nabla f \cdot \boldsymbol{w}
$$

**Intuition:**

If the gradient is a vector that tells us the direction of the steepest ascent, and we want to know how much of this change is in the direction of a vector $$\boldsymbol{w}$$. 

Therefore, if we first compute the gradient of $f$, we can then se how much each vector component influence the function by minor changes.

$$
\nabla_{\boldsymbol{w} }f(x,y)= w_1 \frac{\partial f}{\partial x} + w_2 \frac{\partial f}{\partial y} = \nabla f \cdot \boldsymbol{w}
$$

Important!

Some use the constraint that $$\boldsymbol{w}$$ be a unit vector, this ensures that it is only the directions we are comparing when comparing the directional derivatives. 

In this light, directional derivatives are basically scalar projections between a gradient vector and a unit vector. In any case, if we want to interpret the resulting directional derivative as a slope, we need to convert $$\boldsymbol{w}$$ to a unit vector!

To this end, we need to use the concepts of 1) scalar projection 

The concept falls under projection

<img src="/assets/multical/Untitled 3.png" alt="image-rd" style="zoom:50%;" />

We know from math about rectangular triangles, that:

$$
\begin{align}
\operatorname{cos}(\theta)=\frac{\text{adj} }{\text{hyp} } = \frac{\text{adj} }{\|\boldsymbol{a}\|}
\end{align}
$$

By the same token, we know that $$\boldsymbol{a} \cdot \boldsymbol{b} = |\boldsymbol{a}\|\boldsymbol{b}| \operatorname{cos}(\theta)$$, and hence 

$$\operatorname{cos}(\theta)=\frac{\boldsymbol{a}\cdot\boldsymbol{b} }{\|\boldsymbol{a}\|\boldsymbol{b}\|}$$

Scalar projection

We know that $$\|\boldsymbol{a}\| \operatorname{cos}(\theta) = \text{adj} = \boldsymbol{a}\cdot \frac{\boldsymbol{b} }{\|\boldsymbol{b}\|} = \boldsymbol{a} \cdot \hat{\boldsymbol{b} }$$

This gives us the scalar projection, it provides the magnitude, or the length of the given vector.

Vector projection (bonus)

Now, a vector consist of both a magnitude and a direction. Therefore, to move from a scalar to a vector, we need to multiply this magnitude (i.e. the scalar projection) onto a vector in the same direction of $$\boldsymbol{b}$$. To keep the magnitude intact however, it must be the unit vector, i.e. a vector with the length one. The unit vector of b: $$\hat{\boldsymbol{b} } = \frac{\boldsymbol{b} }{|\boldsymbol{b}|}$$.

Therefore, the vector projection is given by: $$(\boldsymbol{a} \cdot \hat{\boldsymbol{b} })\hat{\boldsymbol{b} } = \frac{\boldsymbol{a}\boldsymbol{b} }{\|\boldsymbol{b}\|} \frac{\boldsymbol{b} }{\|\boldsymbol{b}\|} = \frac{\boldsymbol{a}\boldsymbol{b} }{\|\boldsymbol{b}\|^2}\boldsymbol{b} = \frac{\boldsymbol{a}\boldsymbol{b} }{\boldsymbol{b}\boldsymbol{b} }\boldsymbol{b}$$ .

**The connection to partial derivatives:**

Remember the original form of a partial derivative

$$
\frac{df}{dx}f(a,b) = lim_{h\rightarrow 0} \frac{f(a+h, b)- f(a,b)}{h} \ .
$$

We can rewrite it in vector form

$$
\frac{df}{dx}f(\boldsymbol{a}) = lim_{h\rightarrow 0} \frac{f(\boldsymbol{a}+h\hat{\boldsymbol{i}_1})- f(\boldsymbol{a})}{h} \ ,
$$

where $\hat{\boldsymbol{i}_1} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, i.e. the unit vector in the x direction.

The formal definition of the directional derivative:

$$
\nabla_{\boldsymbol{v} } f(\boldsymbol{a}) = lim_{h \rightarrow 0} \frac{f(\boldsymbol{a} + h\boldsymbol{v}) - f(\boldsymbol{a})}{h}
$$

We can use directional derivatives to prove that the gradient vector point in the direction of steepest ascent.

### Proof that the gradient vector point in the direction of steepest ascent

Let $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ be a differentiable function, and let $$\boldsymbol{u} \in \mathbb{R}^n$$ be a unit vector. The directional derivative of $$f$$ in the direction of $$\boldsymbol{u}$$ at a point $$x$$ is given by:

$$
D_{\mathbf{u} }f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}
$$

Where $$\nabla f(x)$$ is the gradient of $$f$$ at $$x$$ and $$\cdot$$ denotes the dot product. 

*Goal:* 

We want to find the direction $$\boldsymbol{u}$$ that maximises $$D_{\mathbf{u} }f(\mathbf{x})$$.

Using the properties of the dot product, we have: 

$$\nabla f(\mathbf{x}) \cdot \mathbf{u} = \lVert \nabla f(\mathbf{x}) \rVert \lVert \mathbf{u} \rVert \cos(\theta)$$ where $$\lVert \cdot \rVert$$ denotes the magnitude and $$\theta$$ is the angle between $$\nabla f(\mathbf{x})$$ and $$\mathbf{u}$$.
Since $$\mathbf{u}$$ is a unit vector, $$\lVert \mathbf{u} \rVert = 1$$. Thus, the expression becomes: $$\nabla f(\mathbf{x}) \cdot \mathbf{u} = \lVert \nabla f(\mathbf{x}) \rVert \cos(\theta)$$

To maximize $$D_{\mathbf{u} }f(\mathbf{x})$$, we need to maximize $$\cos(\theta)$$. The maximum value of $$\cos(\theta)$$ is 1, which occurs when $$\theta = 0$$. This means that $$\mathbf{u}$$ and $$\nabla f(\mathbf{x})$$ are in the same direction. This ends the proof $$\blacksquare$$

*Conclusion:*

The directional derivative $$D_{\mathbf{u} }f(\mathbf{x})$$ is maximized when $$\mathbf{u}$$ is in the direction of the gradient $$\nabla f(\mathbf{x})$$. Hence, the gradient vector $$\nabla f(\mathbf{x})$$ points in the direction of steepest ascent of the function $$f$$ at the point $$\mathbf{x}$$.
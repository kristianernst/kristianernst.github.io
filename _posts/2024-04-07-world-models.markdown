---
layout: post
comments: true
title:  "On LLMs as world models"
author: "kristian"
excerpt: "learning representations of worlds"
date:   2024-04-07 22:00:00
category: "Philosophy"
mathjax: true
---

I want to spend a few minutes commenting on the notion of "world models", as they have been commonly used descriptors of large language models (LLMs). LLMs are referred to as world models due to the fact that they are incredibly good at storing textual information. As they are trained on vast amounts of human-written text, essentially they are a representation of the world we live in.

Firstly, let us understand what modelling really is about and later on reflect more on this connection. Afterward, I intend to let my thoughts wander unstructured and see if I get somewhere interesting.

### Modelling is about representation

A model is a simplified version of the entity it tries to describe, highlighting a subset of relevant attributes that are helpful for a specific case. Its purpose is therefore to represent the true entity (which can be either physical or abstract) in a particular light.

If we knew everything about everything, models would not be useful?
We use models due to epistemic conditions: we cannot know or take into account all factors; therefore, we need a simplified version that allow us to pay attention to the most important factors contributing to a certain phenomena.

A world model is trying to model the *world*.

Humans have developed various representations of the world. Each of us has a mental image of the world we live in: its physics, the social rules of society, spirituality, etc.

We use our compressed representation of the world as a guide for living.
Instincts might be the first representation of the world: survivability and reproduction. Later, as we navigate in the world, we achieve spatio-temporal information that allow us to revise our world models.

An example of how the world model is used, is that of prediction of future states.
Imagine for a moment that you are walking toward your friend who has been waiting for you at the metro. As he approaches you, based on his momentum and body expression you predict that he is about to hug you, and therefore prepare for a hug.

The ability to perform the hug depends on a representation of among other things physics: the ability for you to move appropriately and time the hug, and social: the cues that call for a hug rather than a handshake, etc.

Much of this information is something we take for granted, however, it is nonetheless there.

### What is text?

Text, in its essence, is a distilled representation of human experience and the physical world. It condenses complex real-world phenomena into symbolic language, which can be communicated and understood without the need for direct experience. For example, when we describe an apple falling to the ground, we're not just recounting an event but invoking the laws of physics—specifically gravity—in a form that can be conceptually and visually understood through words alone.

For LLMs, text is the training substrate. By ingesting vast corpora of text, LLMs are exposed to a broad spectrum of human knowledge and social constructs. Thus, text serves as a proxy for the world in the data these models learn from. It's this foundational role that justifies referring to LLMs as world models: they are trained to simulate understanding and generate outputs based on the wide-ranging human narratives encoded in text.

### Representation squared

The concept of "representation squared" refers to a meta-level of abstraction achieved by large language models.

Initially, the text itself is an abstraction — a representation of the world as humans perceive and describe it. When we train LLMs, we are not merely passing raw data into these models. Instead, we compress and encode this data into a format that the model can process via the embedding layer. Embeddings are dense, continuous vector spaces that capture semantic meanings of words.

These embeddings represent a lower-dimensional manifold, a concept derived from manifold hypothesis in machine learning, which posits that high-dimensional data (like text) tends to lie on or close to a much lower-dimensional manifold. In the case of LLMs, this means the model learns to navigate and generate text based on this compact, but richly informative, semantic space. Therefore, an LLM can be seen as a representation of a representation — first, of the text it trains on, and second, of the underlying concepts and worldviews that the text embodies.

Hence, the LLM itself compresses information in the text into a lower dimensional manifold. The LLM is therefore a representation squared. It is a representation of a representation of the world.

Of course every layer post the embedding layer of the neural network architecture of the LLM is also an abstract representation of the data ...

### Representation common in ML

Representation through compression is a foundational concept in machine learning, not just for processing efficiency but also for enhancing model performance. This principle has been successfully applied across various domains, including recommendation systems, where it has led to significant breakthroughs.

One notable example was during the Netflix Prize, a competition aimed at improving the accuracy of predictions for how much someone is going to enjoy a movie based on their movie preferences. The winning approach leveraged matrix factorization techniques, which are a form of representation compression. This method decomposes a large user-item interaction matrix into lower-dimensional matrices representing latent factors associated with users and movies. By doing so, it was possible to capture deeper, implicit preferences and interactions that are not visible at the surface level.

This approach not only enhanced computational efficiency by reducing the complexity of the data but also improved the accuracy and robustness of recommendations. Traditional methods, which often relied on simplistic rules like recommending the most popular items, lacked the nuance and personalization that matrix factorization provided. The success of this method underscored a key advantage of representation via compression: it helps to filter out noise and enhance the signal, allowing for more precise predictions.

The underlying principle guiding this is closely related to the manifold hypothesis in machine learning, which suggests that real-world high-dimensional data (such as images, text, and user preferences) often lie on or near a much lower-dimensional manifold. This hypothesis implies that even though data might appear complex and high-dimensional, its intrinsic structure can often be represented in much simpler, lower-dimensional terms.

Compression techniques, therefore, do not just simplify data arbitrarily; they seek to uncover and preserve this underlying structure, effectively distilling the essence of the data. This distilled essence is what enables algorithms to perform tasks like prediction, classification, and recommendation more effectively, by focusing on the core features that govern the phenomena being modeled.

The power of representation via compression extends beyond recommendation systems. In natural language processing, techniques such as word embeddings compress vocabulary into dense vectors that capture semantic meanings. In image processing, convolutional neural networks learn to compress visual input into feature maps that emphasize salient features necessary for tasks like object recognition.

Thus, the strategy of creating a simplified model of a pre-existing model is not just a matter of efficiency but is a profound method to enhance the model's interpretability and performance. By reducing complexity and focusing on essential representations, machine learning practitioners can achieve more with less, navigating the trade-offs between accuracy and computational demands adeptly.

### Human and AI representations, comparing the two

TODO:

* adaptive learning
* text vs all sensory/emotional experiences
* differences in loss functions
* biostack, understanding, intuition?

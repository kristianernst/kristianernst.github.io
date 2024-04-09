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

Humans have developed various representations of the world. Each of us has a mental image of the world we live in: its physics, the social rules of society, etc.

We use our compressed representation of the world as a guide for living.
Instincts might be the first representation of the world: survivability and reproduction. Later, as we navigate in the world, we achieve spatio-temporal information that allow us to revise our world models.

### What is text?

Text, in its essence, is a distilled representation of human experience and the physical world. It condenses complex real-world phenomena into symbolic language, which can be communicated and understood without the need for direct experience. For example, when we describe an apple falling to the ground, we're not just recounting an event but invoking the laws of physics in a form that can be conceptually and visually understood through words alone.

For LLMs, text is the training substrate. By ingesting vast corpora of text, LLMs are exposed to a broad spectrum of human knowledge and social constructs. It's this foundational role that justifies referring to LLMs as world models: they are trained to simulate understanding and generate outputs based on the wide-ranging human narratives encoded in text.

### Representation squared

The concept of "representation squared" refers to a meta-level of abstraction achieved by large language models.

The text itself is an abstraction. When we train LLMs, we are not merely passing raw data into these models. Instead, we encode this data into a format that the model can process. The embedding represents the initial layer of abstraction, transforming raw textual input into a structured vector space where semantic similarities and language patterns are mapped into geometrical proximities. 

Throughout the layers of the large language model, these vector representations are transformed into increasingly abstract representations of the data. A parallel to the manifold theory might suggest that each layer of the model attempts to refine the embedding space, moving and shaping it towards a manifold that better represents the complexities of language. Thus, the LLM as it learns, tries to represent the "natural language space" with a lack of better words. 

Approximating / representing this natural language space, which itself can be understood as a representation, leads to the coining: **representation squared**.

LLMs are trying to generate a representation of natural written language: text, and text itself is a representation. It is a representation of a representation of the world.

### Representation common in ML

Representation through compression is a foundational concept in machine learning, not just for processing efficiency but also for enhancing model performance. This principle has been successfully applied across various domains, including recommendation systems, where it has led to significant breakthroughs.

One notable example was during the Netflix Prize, a competition aimed at improving the accuracy of predictions for how much someone is going to enjoy a movie based on their movie preferences. The winning approach leveraged matrix factorization techniques, which are a form of representation compression. This method decomposes a large user-item interaction matrix into lower-dimensional matrices representing latent factors associated with users and movies. By doing so, it was possible to capture deeper, implicit preferences and interactions that are not visible at the surface level.

This approach not only enhanced computational efficiency by reducing the complexity of the data but also improved the accuracy and robustness of recommendations. Traditional methods, which often relied on simplistic rules like recommending the most popular items, lacked the nuance and personalization that matrix factorization provided. The success of this method underscored a key advantage of representation via compression: it helps to filter out noise and enhance the signal, allowing for more precise predictions.

The underlying principle guiding this is closely related to the manifold hypothesis in machine learning, which suggests that real-world high-dimensional data (such as images, text, and user preferences) often lie on or near a much lower-dimensional manifold. This hypothesis implies that even though data might appear complex and high-dimensional, its intrinsic structure can often be represented in much simpler, lower-dimensional terms.

Compression techniques, therefore, do not just simplify data arbitrarily; they seek to uncover and preserve this underlying structure, effectively distilling the essence of the data. This distilled essence is what enables algorithms to perform tasks like prediction, classification, and recommendation more effectively, by focusing on the core features that govern the phenomena being modeled.

The power of representation via compression extends beyond recommendation systems. In natural language processing, techniques such as word embeddings compress vocabulary into dense vectors that capture semantic meanings. In image processing, convolutional neural networks learn to compress visual input into feature maps that emphasize salient features necessary for tasks like object recognition.

Thus, the strategy of creating a simplified model of a pre-existing model is not just a matter of efficiency but is a profound method to enhance the model's interpretability and performance. By reducing complexity and focusing on essential representations, machine learning practitioners can achieve more with less, navigating the trade-offs between accuracy and computational demands adeptly.

### If we all were to die, I think I would preserve LLMs (or the internet) for extraterrestrial beings to learn about the human race.

Weird parallel but I believe it puts thing into perspective. LLMs learns about the world as we see it and in effect about the human race.

We model LLMs to learn about our world as we see it.
It functions as a comprehensive knowledge repository that has a great UI for acquiring knowledge about us.


### Human and AI representations, comparing the two

TODO:

* adaptive learning
* text vs all sensory/emotional experiences
* differences in loss functions
* biostack, understanding, intuition?

---
layout: post
comments: true
title:  "SD - Building a stable diffusion from scratch in PyTorch"
excerpt: "SD."
date:   2023-08-20 22:00:00
category: "SD"
mathjax: true
---

# Building a stable diffusion from scratch in PyTorch

These notes are taken from the video: [link](https://www.youtube.com/watch?v=ZBKpAp_6TGI) by Umar Jamil

# What is stable diffusion?

A model introduced in 2022, by stabilityAI

General usecase: text-to-image, image-to-image, etc.



# Generative model

A model that learns a probability distribution such that we can sample from the distribbution to create new instances of data.

Example: If we have many samples of a cat, then we can train a generative model of it to sample from this distribution to create new images of the cat.

*Why model data as distributions:* we create a very complex where each pixel is a distribution, the entirety of pixels in the image is a complicated joint distribution. we thereby generate samples from it

**Goal**: learn complex distribution and sample from it

Reverse process:
image -> add noise -> add more noise -> until z_t, where we have complete noise.

There are some overlap with the variational autoencoder ([Autoencoders](../architectures/autoencoders.md))



# Denoising diffusion

[Link to notes](./denoising_diffusion_probabilistic_models.md)

Basically we add noice to an image, and learn a neural net to reverse this noice. We condition this learning process by a given transition state \\(t\\) (noice level) of the image \\(\textbf{x}_t\\) and the original image \\(\textbf{x}_0\\).

<img src="/assets/sd/image-20231007155818244.png" alt="image-20231007155818244" style="zoom:28%;" />



How do we do it on a practical level: 

After we get the ELBO, we can parameterize the loss as the following. 
\\(\\)
\nabla\theta \left \| \boldsymbol\epsilon - \boldsymbol\epsilon_\theta \left( \sqrt{\bar{\alpha}_t}\textbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},t\right)\right \|^2
\\(\\)
We need to train a network called \\(\boldsymbol\epsilon_\theta\\) , that, given a noisy image \\(\left( \sqrt{\bar{\alpha}_t}\textbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},t\right)\\) (this means that the noisy image at time-step \\(t\\) and the time-step at which the noice was added: \\(t\\)). The network has to predict how much noise is in the image. If we do gradient descent on (1) we will maximize the ELBO and correspndingly we will maximize the log-likelihood of our data: \\(\log p_\theta(\textbf{x}_0)\\).

*How do we generate new data?*

- We start from complete noise \\(\textbf{x}_T\\) and ask how much noise is there? then we remove it and move to \\(\textbf{x}_{T-1}\\)
- We iteratively do this until we get to \\(\textbf{x}_{0}\\)
- Therefore, IF we start from pure noise, we can generate new images.

We want to control the notification process: if we want to generate a picture of a cat, we cant really start from completely pure noise.

- We need to find a way to tell the model what we want.
- We start from pure noise, but we introduce a signal: call it a prompt, or a conditioning signal in which we influence the model to how to remove the noise.

How can we do this?

- Well, the loss function is currently solely mimicking the pixel data.

- We need to do something different: one solution is to use a joint distribution between the input data and the conditioning signal: \\(p(\textbf{x}_0, c)\\)

- This is still a bad solution, because we really want to model the underlying distribution \\(p_\theta(\textbf x_0)\\) and the conditioning signal \\(c\\) will likely interfere with this goal.

- The \\(\boldsymbol \epsilon_\theta\\)  will be built using the U-net model architecture.

- The U-net will receive an input. The input is 1) the image at a specific noise level, and 2) the noisification level \\(t\\) of the image.

  - We could extend the input to also include 3) the prompt / conditioning signal. This way, if we tell the model: remove noise from this image, it has this level of noise, and it is an image of a cat. The model will be better at generating an image in alignment with the prompt.
  - It is therefore a way to condition the model.

- But at the same time, instead of including the prompt every time while training, we can say, do it only 50% of the time.

  - This way, the model will learn to act both as a conditional model and also as an unconditional model.
  - The advantage is that we can do the following:
    1. Generate an image based on image 1 at noise level 3 with a prompt
    2. Generate an image based on image 1 at noise level 3 without a prompt
    3. Then we combine these two outputs in such a way that we decide how much we want the image to be closer to the conditioning signal
  - This is called "Classifier free guidance"
  - \\(\text{output} = w \cdot (\text{output}_{\text{conditioned} } - \text{output}_{\text{uonditioned} }) + \text{output}_{\text{conditioned} }\\)
  - By controlling the power of \\(w\\), we can decide how we condition the final image on the prompt.

- The model needs to understand the prompt. This is where CLIP comes in.

  

# CLIP

Clip is short for contrastive language-image pre-training. 

It is a way to introduce prompts into the learning. So for-example, together with an image of a dog, we can include a prompt saying: "an image of a dog." 
This prompt needs to be encoded by a text encoder.



[CLIP](https://openai.com/research/clip) is a model developed by openAI that allows to connect text with images.

How is it done: 
![/assets/sd/overview-a.svg](//assets/sd/sd//assets/sd/overview-a.svg){:height="50%" width="50%"}

Image 1 has a corresponding text (text 1). We train the model to be able to connect the text and image together, i.e. pick the right text for an image and vice versa. This is illustrated by the blue diagonal of the interaction matrix. Hence, the loss function is to maximize the values of \\(I_i T_i\\) and thereby minimize  \\(I_iT_j, i\neq j\\).

==In stable diffusion, we take the text encoder of CLIP to make embeddings of the prompt. We use these embeddings as a conditioning signal to the units of the noice.==



## Reversing the noisified image with the classifier-free guidance

[VAE](../architectures/autoencoders.md)

The reverse process means that we need to take many steps of denoisification. IF the image is very big, it means that every time we will have a very big matrix to work through at each step. This will take a very long time.  What IF we could compress this image into something smaller so each unit takes less time. Here we use the VAE.

Stable diffusion is actually called a latent diffusion model in which we dont learn the distribution of \\(p_\theta(\textbf{x}_0)\\) but rather the distribution of the latent representation of our data using VAE. Then we can decompress the compressed data again at some point.

For general autoencoders the code \\(\textbf{z}\\) of different images cannot be semantically compared. I.e. the code for a cat might look very similar to the code of a pizza slice, because the auto-encoder simply fit the data manifold without any condition. To make codings of different images more discernable and thereby more semantically meaningful, we use variational autoencoders.

We learn to compress the data, but at the same time the data is distributed according to a multivariate distribution (most of the times a Gaussian) with a mean \\(\mu\\) and a variance \\(\sigma^2\\) 

<img src="/assets/sd/image-20231007165226798.png" alt="image-20231007165226798" style="zoom:33%;" />



# The complete architecture: text to image

<img src="/assets/sd/image-20231007165306153.png" alt="image-20231007165306153" style="zoom:33%;" />

1. We give a text input

2. We sample some noise, and compress it by the encoder (VAE) to the latent dimension \\(\textbf{z}\\) 

3. We send the prompt embeddings together with the noise sample, plus information about the noise level \\(t\\) 

   1. The U-Net will iteratively denoise the image until we reach no noise. This is tracked by a scheduler.

4. After the U-net has denoised the image, we take the latent dimension \\(\textbf{z}\\) and run it through the decoder of the VAE to get the final vector-representation of the image.

   

# Image-to-Image

<img src="/assets/sd/image-20231007165954166.png" alt="image-20231007165954166" style="zoom:33%;" />

The more noise we add to the latent, the more freedom we give the model to generate the new image.



# In-painting

<img src="/assets/sd/image-20231007170157939.png" alt="image-20231007170157939" style="zoom:33%;" />

If we want to make some new legs for the dog, we can use in-painting.

# Implementation; reflections

After having built the attention mechanism, the VAE, and the CLIP there are some general insights:

1. VAE, when we do compression of images we generally decrease the number of pixels (making the image smaller), but increase the number of channels, i.e. how much information each pixel has.
2. We do attention on the image: how? 
   1. Each pixel in the image can be encoded / "tokenized" as a vector representation. Just like a pixed can be represented by RGB. 
   2. Then each pixel in the image can be related to each other via the attention mechanism.
3. We use groupnormalization, for images because it does not make sense to normalize across ALL tokens. We assume that pixels near each other are more related and therefore are more appropriate to normalize. 
4. All architectures so far has used the attention mechanism (which we manually built so it is slower than the PyTorch documentations implementation). 



Now, we want to build the U-Net, and to that end we need to connect multiple architectures. The U-net takes both the CLIP encodings as well as the latent representation of the image as input.

==How can we relate these two models to each other? With the use of cross-attention==.

The query is from the first sequence and the key and value from another sequence.










---
layout: post
comments: true
title:  "OG positional encoding"
excerpt: "A simple explanation of the positional encoding used in the transformer architecture."
date:   2023-08-20 22:00:00
category: "NLP"
mathjax: true
---

# OG positional encoding

As presented in “Attention is all you need”, the positional encodings are introduced to the token embeddings via addition. 

Here’s a not optimized code for encoding position, but it clearly show what is happening with each token vector:

```python
import torch

# model_dimensionality
emb_size = 4

# the input shape
sequence_length = 4

# init an empty matrix
matrix_token_pos_emb = torch.zeros(sequence_length, emb_size)

# get token positions
tok_position = torch.arange(0, sequence_length)

# now we have what we need;
n = 10000
for p, pos in enumerate(tok_position):
    k = 0
    for i in range(emb_size // 2):
        matrix_token_pos_emb[p, k] = torch.sin(pos * (1 / (n ** (2*i / emb_size))))
        k+=1
        matrix_token_pos_emb[p, k] = torch.cos(pos * (1 / (n ** (2*i / emb_size))))
        k+=1

print(matrix_token_pos_emb)
> tensor([[ 0.0000,  1.0000,  0.0000,  1.0000],
          [ 0.8415,  0.5403,  0.0100,  0.9999],
          [ 0.9093, -0.4161,  0.0200,  0.9998],
          [ 0.1411, -0.9900,  0.0300,  0.9996]])
```

**So what is going on?**

Well, it is clear by looking at the inner loop of the code, that we leverage the sine and cosine to generate positional encodings. The sines are used on the even elements of the token vector and the cosines are used of the uneven elements.

Also, we keep track of the “position” of each token and multiply this position (integer) with some division term. This division term takes a parameter \\(n\\) (\\(=10000\\) in the original implementation)  and scales this by some power. It is clear that this scaling is dependent on \\(i\\), that is to say, an index that only gets updated once we have stored two values in our `matrix_token_pos_emb`. 

Why? because we essentially decompose every token vector into two parts: one vector containing the values of even positions, and one containing the values of uneven positions. The index i, is therefore used twice before it is updated. eventually we combine the two vectors again to regain the same length as the positional encoding vector: [even 0, uneven 0, even 1, uneven 1] in our case.

Here is the mathematical formula that essentially tells the same thing (I found the updating of i unintuitive, in the formula):

$$
PE _{(pos, 2i)} = sin\left(pos/n^{2i/d_{model} }\right) \\ 
PE _{(pos, 2i + 1)} = cos\left(pos/n^{2i/d_{model} }\right)
$$

So now that we know what is going on, it is time to explain why this makes sense.

This video (10 min) provides a super good explanation: [https://www.youtube.com/watch?v=dichIcUZfOw](https://www.youtube.com/watch?v=dichIcUZfOw)

Positional encoding in general is about storing information about tokens positions in a way that will benefit the model. By “benefit”, I mean that the model is better at discerning which token is which whilst at the same time not impacting predictive performance of the model.

- One way to do positional encoding is simply just to add the integer number of the position of the token in the sequence.
	- This is not a good idea tho! why? because one token position is 0 and another is 2048 (if the sequence length is 2048).  When one combines the regular token embeddings with this positional encoding, embeddings later in the sequence will inflate and therefore only bring noise to the network.
	- We could also just scale everything to range between 0 and 1, however, this would not really benefit either. for one, it gives more importance to the later tokens in the sequence. Also, if sequence length varies, these calculations vary which results in the same position being computed differently at different time-steps.

Okay, so we like an encoding architecture that: 

1. allow the model to discern between tokens
2. compute the position of a token the same way independent of sequence length

Although the sample we provided in the code expression far from justifies a proof, we can see that each token vector is different from another. This is good!

The reason this works, is that the model leverages different frequencies of the sine and cosine waves. 

- high-frequency components help the model discern tokens that are near each other
- low-frequency components help the model discern tokens far from each other.

This could really be done using only sine or cosine, not necessarily both.

- However, since sine and cosine are orthogonal to each other, including both dramatically increases information that the model can use to discern between tokens.
- Furthermore, a [proof](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/) exist that \\(PE_{pos + k}\\) can be represented as a linear function of \\(PE_{pos}\\). It has been shown when we utilize the switching between cos and sine. But as far as I can tell, it has not been shown using only one or the other ([https://kazemnejad.com/blog/transformer_architecture_positional_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)).
	- Why is this property nice?
		- It is nice, because it means that a model can generalize well to new sequence lenghts, because it can grasp this linear relationship.
		- We therefore increase the model’s generalization capabilities.

## Simple intuition behind the claim: Linearity of Positional Encoding

### Definitions

First, let's define the positional encoding \\(PE\\) as given in the original "Attention Is All You Need" paper:

For even  \\(i\\):

\\(PE(p, i) = \sin\left(\frac{p}{10000^{\frac{2i}{d} } }\right)\\)

For odd \\(i\\):

\\(PE(p, i) = \cos\left(\frac{p}{10000^{\frac{2i}{d} } }\right)\\)

### Goal

Our goal is to show that  \\(PE(p+k)\\) can be represented as a linear function of \\(PE(p)\\) for any fixed offset \\(k\\).

### Proof for Even  \\(i\\)

### Step 1: Define \\(PE(p+k, i)\\)

First, let's define the positional encoding at \\(p+k\\) for an even \\(i\\):
\\(PE(p+k, i) = \sin\left(\frac{p+k}{10000^{\frac{2i}{d} } }\right) = \sin \left( \frac{p}{ {10000^{\frac{2i}{d} } } } + \frac{k}{ {10000^{\frac{2i}{d} } } } \right)\\)

### Step 2: Use the Angle Addition Formula for Sine

We can rewrite \\(PE(p+k, i)\\) using the angle addition formula for sine:

\\(\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)\\)

Applying this to \\(PE(p+k, i)\\), we get:

\\(PE(p+k, i) = \sin\left(\frac{p}{10000^{\frac{2i}{d} } }\right)\cos\left(\frac{k}{10000^{\frac{2i}{d} } }\right) + \cos\left(\frac{p}{10000^{\frac{2i}{d} } }\right)\sin\left(\frac{k}{10000^{\frac{2i}{d} } }\right)\\)

### Step 3: Relate to \\(PE(p, i)\\)

Notice that this expression is a linear combination of \\(PE(p, i)\\) and \\(PE(p, i+1)\\):
\\(PE(p+k, i) = PE(p, i)\cos\left(\frac{k}{10000^{\frac{2i}{d} } }\right) + PE(p, i+1)\sin\left(\frac{k}{10000^{\frac{2i}{d} } }\right)\\)

### Proof for Odd \\(i\\)

The proof for odd \\(i\\) is similar and uses the angle addition formula for cosine:
\\(\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)\\)

### Conclusion

Thus, for any fixed offset \\(k\\),  \\(PE(p+k)\\) can be represented as a linear function of \\(PE(p)\\). This property is crucial for the model's ability to generalize to sequence lengths it hasn't seen during training.
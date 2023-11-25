---
layout: post
comments: true
title:  "Make Pytorch fast"
excerpt: "import torch"
date:   2023-08-20 22:00:00
category: "PyTorch"
mathjax: true
---

### Data loading

#### Batches

#### Multiprocessing

---

### Model training

---

#### Data parallelization techniques

All techniques generally follow the PyTorch documentation [Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)

There are generally three steps which a model iterates over while training:

1. Forward pass (run model and get output)
2. Backward pass (evaluate loss and calculate gradients)
3. Optimize (Update parameter weights based on gradients and optimizer)

- **DP (data parallel)**

	Data parallel essentially distributes work from one GPU to multiple other GPU’s

	Let’s call GPU_0 the main GPU and GPU_1 , … GPU_n the GPUs that are allocated work

	1. In the forward pass GPU_0 replicates its model parameters, settings, etc to allocated worker GPUs. Assume we have four GPUs and a batch size of 30, in this case GPU_0 divides the batches into three equal parts each forward pass (GPU_1 to GPU_3 each get a batch_size of 10). 
	2. During the backward pass, the loss and gradients are computed for each worker GPU. These gradients are then summed to the original module (GPU_0).
	3. The model parameters are then updated in GPU_0. We repeat process 1 - 3 until the model has finished training.

	Drawbacks: 

	We spend much time communicating between GPU_0 and the allocated worker GPUs.

	We iteratively replicate GPU_0’s model each epoch which is taking up memory and computation

- **DPP (Distributed data parallel)**

	Distributed data parallel generally works by dividing the entire model training on multiple GPUs from the get-go. 

	1. Each GPU runs a model which is initialized by the same random_seed, (then forward pass is computed in parallel). Each Model is always fed a unique subset of the data input while training.
	2. The loss and gradients are calculated independently for each model on each GPU. Then the gradients are updated using an “AllReduce” communication between the different GPUs. While different AllReduce operations exist, the main takeaway is that it effectively “syncs” the gradients in each model.
	3. The models are then updated based on these synced gradients to ensure that all models are equivalent. We then repeat steps 1-3 till we have finished training.

- **Sharding**

	Sharding is somewhat similar to `DPP`, however, it removes the redundancies resulting from the fact that every GPU maintains a copy of all optimizer states, forward and backward activations. Each GPU only stores a subset of these informations thereby drastically reducing memory usage.

	Read this [article](https://towardsdatascience.com/sharded-a-new-technique-to-double-the-size-of-pytorch-models-3af057466dba) about sharding and a pytorch-lighting implementation 

### Inference

#### Quantization

Quantization means that we perform all ops on integers instead of floating point values. By doing this, we discretize continous signals, which free up memory consumption and also increase transfer speed. 

**Why do we get better speed and memory performance?**

1. Hardware has recently been specialized to do integer operations. 
2. Neural networks are often bottlenecked, not by how many computations they need to do, but by how fast they can transfer data (i.e. the memory bandwidth and cache)
3. transitioning to 8-bit ops (integers) rather than 32-bit ops (floats) means we move data 4x faster.
4. Storing models in integers instead of floats save us approximately 75% of the ram/harddisk space whenever we save a checkpoint. This is especially useful in relation to deploying models using docker (as you hopefully remember) as it will lower the size of our docker images.

**How do we quantize a model?**

$$
x_\textsf{int} = \operatorname{round}\left(\frac{x_\textsf{float} }{\text{scale} }+\text{zero\_point}\right)
$$

Where \\(s\\) is a scale and \\(z\\) is the zero point.

This function is called a “Linear affine quantization”. 

<img src="/assets/pytorch/image-20231005162335159.png" alt="image-20231005224728442" style="zoom:40%;" />

[devblog.pytorchlightning.ai](https://devblog.pytorchlightning.ai/how-to-train-edge-optimized-speech-recognition-models-with-pytorch-lightning-part-2-quantization-2eaa676b1512)

Here is a quick script that demonstrates how the quantization is done:

```bash
>>> import torch
>>> x_float = torch.rand(5,5)*20 - 10
>>> print(x_float)
tensor([[-6.4938,  7.5907,  7.4472, -2.5390,  5.1757],
        [-6.4978, -3.0276,  7.2343,  6.0056, -9.1232],
        [ 5.4574, -5.8731, -6.1708,  7.3719, -0.0546],
        [ 0.1817, -0.1734, -4.3979,  8.4719, -6.0725],
        [-3.4074,  1.9902, -0.0503,  3.0557, -7.3558]])

# define parameters
>>> num_levels = 256 # 2**8 (8bit int, rounding to either 1 or 0)
>>> min_val = x_float.min().item()
>>> max_val = x_float.max().item()
>>> scale = (max_val - min_val) / (num_levels - 1)
>>> zero_point = - min_val / scale
>>> zero_point = round(zero_point)

# quantize
>>> x_int = torch.round(x_float / scale + zero_point)
>>> print(x_int)
tensor([[ 38., 242., 240.,  95., 207.],
        [ 38.,  88., 237., 219.,  -0.],
        [211.,  47.,  43., 239., 131.],
        [135., 129.,  68., 255.,  44.],
        [ 83., 161., 131., 176.,  25.]])
>>> x_int = x_int.to(torch.uint8)
>>> print(x_int)
tensor([[ 38, 242, 240,  95, 207],
        [ 38,  88, 237, 219,   0],
        [211,  47,  43, 239, 131],
        [135, 129,  68, 255,  44],
        [ 83, 161, 131, 176,  25]], dtype=torch.uint8)

# dequantize
>>> x_dequantized = (x_int.float() - zero_point) * scale
>>> print(x_dequantized)
tensor([[-6.4860,  7.5900,  7.4520, -2.5530,  5.1750],
        [-6.4860, -3.0360,  7.2450,  6.0030, -9.1080],
        [ 5.4510, -5.8650, -6.1410,  7.3830, -0.0690],
        [ 0.2070, -0.2070, -4.4160,  8.4870, -6.0720],
        [-3.3810,  2.0010, -0.0690,  3.0360, -7.3830]])

>>> print(x_dequantized - x_float)
tensor([[ 0.0078, -0.0007,  0.0049, -0.0140, -0.0007],
        [ 0.0117, -0.0084,  0.0107, -0.0025,  0.0151],
        [-0.0064,  0.0081,  0.0298,  0.0111, -0.0144],
        [ 0.0253, -0.0336, -0.0181,  0.0151,  0.0005],
        [ 0.0264,  0.0109, -0.0187, -0.0197, -0.0273]])

# torch version (we see that the quantized version is represented as floats when printed)
>>> x_qtorch = torch.quantize_per_tensor(input = x_float, scale = scale, zero_point = zero_point, dtype = torch.quint8)
>>> print(x_qtorch)
tensor([[-6.4860,  7.5900,  7.4520, -2.5530,  5.1750],
        [-6.4860, -3.0360,  7.2450,  6.0030, -9.1080],
        [ 5.4510, -5.8650, -6.1410,  7.3830, -0.0690],
        [ 0.2070, -0.2070, -4.4160,  8.4870, -6.0720],
        [-3.3810,  2.0010, -0.0690,  3.0360, -7.3830]], size=(5, 5),
       dtype=torch.quint8, quantization_scheme=torch.per_tensor_affine,
       scale=0.06900028901941636, zero_point=132)
```

- **Brief information on number of bits and scaling used**

	Remeber a bit is the smallest unit of data in a computer. It has one of two values: 0 or 1. 

	1 bit quantization: values are either 0 or 1

	2 bit quantizaiton: values are either 00, 01, 10, or 11

	…

	8 bit quantization: values can take 256 possible states

	Pattern: \\(2^n\\) where \\(n\\) is the number of bits.

	Therefore, the scaling parameter takes the maximum spread in the tensor and scales it to a (lower) representation, in this case we use downscale from 32 bit to 8 bit. 

#### Model distillation (knowledge distillation)

Here, we train a model (teacher) which is then used to train another model (student).

Hence, the transmitting of knowledge from a rigorous model to a simpler one follows exactly the principle of compression. 

By itself, knowledge distillation has not proved as effective as quantization, but together it has proven more effective
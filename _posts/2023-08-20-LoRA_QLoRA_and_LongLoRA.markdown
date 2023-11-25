---
layout: post
comments: true
title: "Innovations in Fine-Tuning Large Language Models: LoRA, QLoRA, and LongLoRA"
excerpt: "A comprehensive overview of LoRA, QLoRA, and LongLoRA techniques for memory-efficient fine-tuning of LLMs."
date: 2023-08-20 22:00:00
category: "NLP"
mathjax: true
---

# LoRA, QLoRA, and LongLoRA

These two papers have been applied to make the fine-tuning of LLMs cheap in memory and compute. 

Snippet of [LoRA](https://arxiv.org/pdf/2106.09685.pdf) abstract:

 

> As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

Snippet of [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) abstract:

> We present QLORA, an efficient finetuning approach that reduces memory us- age enough to finetune a 65B parameter model on a single 48GB GPU [Normally requires more than 780 GB of GPU memory with regular 16-bit] while preserving full 16-bit finetuning task performance. QLORA backpropagates gradi- ents through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).

Snippet of LongLoRA:

> In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although *dense global* attention is needed during inference, fine- tuning the model can be effectively and efficiently done by *sparse local* attention. The proposed shift short attention (S2-Attn) effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only *two lines of code* in training, while being optional in inference.

So motivations? 

- It is inconvenient to spend a lot of resources on fine-tuning LLMs with the current rapid development of pre-trained LLMs

[LoRA](LoRA.md)

[QLoRA] # todo, write

[LongLoRA](LongLoRA.md)
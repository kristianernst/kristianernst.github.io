---
layout: post
commets: true
title: "Notes on lets verify step by step"
excerpt: "Notes on lets verify step by step"
date: 2023-11-27 00:00:00
category: NLP
---


## The paper

[link](https://arxiv.org/abs/2305.20050)

Background and motivation
- Hallucination: generate text that is not supported by the input
- One point of failure will cause later generations to be wrong (especially true in math problems)
- To avoid this, we need to verify the intermediate steps

*Existing work*:

Reward models trained to discriminate between desirable and undesirable outputs. RMs are then used in:
1. an RL pipeline, 
2. used to perform search via rejection sampling. 


**RL pipeline**: 
- I understand this paper reference the [RLHF](https://arxiv.org/abs/2203.02155) paper. 


**Rejection sampling** [Training verifiers to solve math problems](https://arxiv.org/pdf/2110.14168.pdf): 

The abstract clealy states:
> To increase per- formance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that ver- ification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline.

So generally, it is an approach that involves generating a large number of samples from a model and then filtering these samples according to certain criteria to find the ones that best meet the desired objectives.

They did an interesting thing in this paper: 

They compared the performance by autoregressively sampling a single low temp solution and checking whether the final answer is correct, to sampling multiple high temperature solutions assigning each solution with a score and outputting the highest ranked solution. The verification method is trained to judge the best solution simply with respect to whether or not the solution reached the final answer.

A way to increase creativity, and reasoning?

How the verifier is trained:

*Auxiliary Objective for the Verifier*: Here, the verifier model, which is primarily designed to assess the correctness of solutions, is also trained with a language modeling objective. This means that in addition to its main task of verifying solutions, the verifier is also trained to predict the next word in a sequence, which is a typical language modeling task. This secondary task serves as an auxiliary objective. Training the verifier on this auxiliary language modeling objective helps improve its understanding of language and context, which in turn can enhance its primary function of verifying solutions.

*Auxiliary Signal in Token-Level Verifying*: In the second instance, the token-level verifier (which assesses the value or correctness of individual tokens in a sequence, rather than the entire solution) is said to benefit from the "full value function" as an auxiliary signal. This implies that by evaluating each part of a solution (each token), the model receives additional, helpful information that aids in its overall task. This more granular approach to verification provides the model with a richer set of data to learn from, encouraging it to understand the reasoning process throughout a solution, rather than just focusing on the end result. This auxiliary signal, therefore, helps the model to avoid overfitting and to maintain its ability to generalize, as it must consider the entire reasoning process, not just the final answer.


**Back to the paper at hand**

Lets verify step by step uses a fixed generator model, which is a finetuned GPT 4 base model, without RLHF. 

It is trained on a PRM800K dataset, a curated dataset.

The paper evaluates Reward Models solely, i.e. it does not use RL afterwards. 

It evaluates the performance of Outcome-supervised Reward Models (ORMs) and Process-supervised Reward Models (PRMs)

Quote about the ORMs: 
> We uniformly sample a fixed number of solutions per problem from the generator, and we train the ORM to predict whether each solution is correct or incorrect. In practice, we usually determine correctness by automatically checking the final answer, but in principle these labels could be provided by humans. At test time, we use the ORM’s prediction at the final token as the overall score for the solution. We note the automatic grading used to determine ORM targets is not perfectly reliable: false positives solutions that reach the correct answer with incorrect reasoning will be misgraded.

PRMs:
> train to predict the correctness of each step after the last token in each step. This prediction takes the form of a single token, and we maximize thelog-likelihood of these target tokens during training. The PRM can therefore be trained in a standard language model pipeline without any special accom- modations. To determine the step-level predictions at test time, it suffices to perform a single PRM forward pass over the whole solution.
> To compare multiple solutions, it is necessary to compute a single score for each solution. This is an important but straightforward detail: we define the PRM score for a solution to be the probability that every step is correct under the PRM. We implement this as the product of the correctness probabilities for each step.


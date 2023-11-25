---
layout: post
comments: true
title:  "GPT-Fathom"
excerpt: "A detailed look into GPT-Fathom"
date:   2023-08-20 22:00:00
category: "NLP"
mathjax: true
---

# GPT-Fathom

[https://arxiv.org/pdf/2309.16583.pdf](https://arxiv.org/pdf/2309.16583.pdf)

- Many existing LLM leaderboards reference scores from other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. To achieve reliable conclusions, it is crucial to make apples-to-apples LLM comparisons with consistent settings and prompts.
- Many existing works assess LLMs on merely one or a few aspects of capabilities, which is not sufficient to provide a comprehensive view to deeply understand the strength and weakness of the evaluated LLMs
- Many existing works only focus on the benchmark score under one specific setting, while overlooking the impacts of model sensitivity on the overall usability of LLMs. In fact, it is unacceptable that a slightly rephrased prompt could cause the LLM to fail in responding it correctly.

**Problem:**

These challenges hinder a comprehensive understanding of LLMs. 

**Action:**

To dispel the mist among LLM evaluations, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite de- veloped based on OpenAI Evals1. We evaluate 10+ leading open-source and closed-source LLMs on 20+ curated benchmarks in 7 capability categories under aligned settings.

Side-bonus:

The authors trace the progressive development from GPT-3 to GPT-4, aiming to help the community understand its “enigmatic path”

**Current benchmarks:**

There are many great existing LLM evaluation suites. By comparing GPT-Fathom with previous works, we summarize the major difference as follows: 1) HELM (Liang et al., 2023) primarily uses answer-only prompting (without CoT) and has not included the latest leading models such as GPT-4 (as of the time of writing); 2) Open LLM Leaderboard (Beeching et al., 2023) focuses on open-source LLMs, while we jointly consider leading closed-source and open-source LLMs; 3) OpenCompass (Contributors, 2023) evaluates latest open-source and closed- source LLMs (all released after 2023/03), while we cover both leading LLMs and OpenAI’s earlier models to decipher the evolutionary path from GPT-3 to GPT-4; 4) InstructEval (Chia et al., 2023) is designed for evaluating instruction-tuned LLMs, while we evaluate both base and SFT / RLHF models; 5) AlpacaEval (Li et al., 2023) evaluates on simple instruction-following tasks as a quick and cheap proxy of human evaluation, while we provide systematic evaluation of various aspects of LLM capabilities; 6) Chatbot Arena (Zheng et al., 2023) evaluates human user’s dialog preference with a Elo rating system, while we focus on automatic and reproducible evaluation over popular benchmarks; 7) Chain-of-Thought Hub (Fu et al., 2023) focuses on evaluating the reasoning capa- bility of LLMs with CoT prompting, while we support both CoT and answer-only prompting settings and evaluate various aspects of LLM capabilities.

**Benchmarking benchmarks**

We consider the following criteria for benchmark selection: 1) cover as many aspects of LLM capabilities as possible; 2) adopt widely used benchmarks for LLM evaluation; 3) clearly distinguish strong LLMs from weaker ones; 4) align well with the actual usage experience of LLMs. Accord- ingly, we construct a capability taxonomy by initially enumerating the capability categories (task types), and then populating each category with selected benchmarks.

The benchmarks are:

- Reasoning: measures performance on common sense tasks as well as comprehensive reasoning tasks.
- Comprehension: assesses the reading comprehension ability of the models
- Math: scores the mathematical capabilities of a model
- Coding: evaluates coding capabilities of the model
- Multilingual: measures the models ability to translate, but also how well the model performs in terms of reasoning, comprehension, math, etc. in different languages
- Safety: scores the models ability to generate truthful, non-harmful, non-biased content.
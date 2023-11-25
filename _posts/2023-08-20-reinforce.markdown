---
layout: post
comments: true
title:  "Reinforce"
excerpt: "Insights into RL"
date:   2023-08-20 22:00:00
category: "RL"
mathjax: true
---

## Vocabulary



| Symbol     | Name                         | Definition                                 |
| ---------- | ---------------------------- | ------------------------------------------ |
| $$s$$        | state                        | environment state: $$s \in S$$               |
| $$a$$        | action                       | $$a \in A$$                                  |
| $$s_{t+1}$$  | transition                   | $$\operatorname{Env}(s_t, a_t)$$             |
| $$r_{t+1}$$  | reward                       | $$\operatorname{Reward}(s_t, a_t, s_{t+1})$$ |
| $$o_t$$      | observation                  | $$o_t = (r_t, s_t)$$                         |
| $$\pi$$      | policy                       | distribution over $$a$$ given a history      |
| $$R$$        | cumulative discounted reward | $$R = \sum_{t=1}^\infty \gamma^{t-1} r_t$$   |
| $$\rho^\pi$$ | expected cumulative reward   | $$\rho^\pi = \mathbb{E}[R|\pi]$$ .             |



Different approaches to reinforcement learning:

**Brute force**:

Go through all possible policies $$\pi$$ and estimate $$\rho^\pi$$ by sampling

**Value function approaches**:

Value function: $$V^\pi(s)=\mathbb{E}[R|s,\pi]$$

We specify the expected reward both on policy and initial state. If we go one step further, we can also make this dependent on the state we are at, the action we will take at this state and the policy we intend to follow. 

Q-function: $$Q^\pi(s,a) = \mathbb{E}[R|s,a,\pi]$$

There are different approaches of how to optimize these value functions.



**Monte Carlo methods**

Policy estimation - estimate $$Q^\pi(s,a)$$

Policy improvement - update $$\pi$$ based upon estimate.

Think of it as a monte carlo search



**Temporal difference methods**

Change policy on the fly



**Direct policy search**

Make parameterized model for polixy $$\pi_\theta$$

Optimize $$\rho^{\pi_\theta}=\mathbb{E}[R|\pi_\theta]$$ wrt $$\theta$$

You can think of this as a neural network that implements a conditional probability of the different actions (like a softmax probability to a discrete set of actions, given a state / history of state). the weights of the network is then $$\theta$$.

The idea now is that you can try to make an estimate of the expected cumulative reward and take the gradient of this function.

Gradient based: REINFORCE algorithm

Not gradient based (evolutionary): NEAT algorithm



## AlphaGo case

Value function $$v^*(s)$$ outcome for optimal play

s = state (game position)

b = game breadth (different moves available at a given time)

d = game depth (moves needed to terminate the game)

complexity = $$b^d$$ for calculating $$v^*(s)$$

for Chess: b = 35, and d = 80 (apprx.)

for Go: b = 250, d = 150 (apprx.)



Answer: we cannot use brute force to solve the game!

Different ways: 

**Supervised learning policy network**

Let us use a large database of expert player moves: if we can build a network that can imitate pro's then we are far?

- A purely supervised approach. 
- Google needed this as a foundation.

Once you have a somewhat decent player, you can let the model play against itself.

**Reinforcement learning policy network**

- The simulating of games where the network plays against itself

**Value network**

We are also interested in having a network that allow us to look at the board position and provide an estimate of the likelihood that white will win the game



The policy networks takes the board position at a certain time instance, it takes it as an input to a neural network. This neural network will calculate the probability for each possible action.

On the other hand, the value network only produces one number: the probability of white winning given the current board position.



## The four steps in the DeepMind solution

### SL policy

a = action

s = state

$$p_\sigma(a|s)$$ = policy



We have a large database of expert games

We train the classifier to imitate expert moves (s, a):

$$\Delta \sigma \propto \frac{\partial \log p_{\sigma}(a|s)}{\partial \sigma}$$

We want the network to put a high probability to use the moves that the experts used.

### Step 2: reinforcement learning

The policy network ($$p_\rho(a|s)$$) plays against a younger version of itself

-  Record whether it wins/losses: $$Z_t = r(T) = +1 / -1$$
- Train classifier to imitate expert moves $$(s, a)$$:
      $$\Delta \rho \propto \frac{\partial \log p_{\rho}(a_t|s_t)}{\partial \rho} Z_t$$
- Better than SL policy.

We have the derivative of the log probability of the actions, just like before. BUT the term is multiplied with the Z score, which is 1 if it is a win and -1 if it is a loss, thereby reinforcing choices if they lead to wins.

Yields a stronger player than a SL player. 

We cannot win against the world champion with this strategy alone. 

We need to utilize the fact that we can look ahead into the game. Thereby we can roll out the game and make more informed decisions.

The closer we are to the end of the game, the more accurate the estimates of win/loss are.

#### On Value Functions and Reinforcement Learning

Okay, here's the thing: In the world of reinforcement learning (RL), the **value function** is like the Holy Grail. Think of it as your GPS in the vast land of state space, telling you how "good" or "valuable" a certain state is in terms of future rewards.

The ideal value function, `v*`, represents the true expected return of each state. However, and here's the kicker, this bad boy is often unknown. So what do we do when faced with the unknown in the world of AI? We approximate!

##### Approximating the Value Function

The goal is to try to learn an approximation of this value function using a neural network. Let's call this approximation `vθ(s)`. The subscript θ here represents the parameters of the neural network. So, in essence, our RL agent is trying to make its own GPS by training this network.

Now, for the cool part. To train this network, we use the following gradient:
$$
Δθ \propto \frac{∂\log v_θ(s)}{∂θ} (z - v_θ(s))
$$


$$\frac{∂\log v_θ(s)}{∂θ}$$ is the gradient of the log of our approximated value function with respect to the network parameters. It helps us understand how tweaking the parameters affects our approximation.

 is the difference between some target value `z` and our current approximation `vθ(s)`. Think of this as the error in our approximation. We want to minimize this error, right? By multiplying our gradient with this error, we guide our network in the right direction.

##### Using this loss as an ingredient in Monte Carlo tree searches

Now, the real magic happens when we mix this with Monte Carlo tree search (MCTS). MCTS is a clever way to make decisions in complex environments. It's like playing out many potential futures in your mind, and then making a move based on which future looks the brightest.

We can use our approximated value function as an ingredient in this search process. Instead of blindly exploring potential futures, we use `vθ(s)` to score positions `s`. This gives our MCTS a sense of direction, helping it prioritize which futures are worth exploring more deeply.

##### Monte carlo tree search

Monte Carlo Tree Search is a powerful algorithm, particularly when combined with deep learning techniques. Here's a breakdown of the process you see:

1. **Selection**: Starting from the root node (the current state of the game), it selects optimal child nodes until it reaches a leaf node. The selection is based on a combination of the value `Q` (estimated value of the action leading to the node) and an exploration term `u(P)`. The aim is to balance between exploring new nodes (actions) and exploiting the ones with high expected rewards.
2. **Expansion**: Once a leaf node is reached, one or more child nodes are added to expand the search space.
3. **Evaluation**: The evaluation of the newly expanded node is performed using a neural network (in this case, the value network `vθ`). It gives an estimate of the expected outcome from that position.
4. **Backup**: The evaluation of the node is then backed up through the path used in the selection phase to update the values of all traversed nodes. This ensures that the tree's evaluations are consistently updated based on new explorations.

###### Policy and Value Networks:

- The **SL (Supervised Learning)** policy network is used to propose moves in the tree search. This means that the network, trained on expert games, suggests possible next moves.

  It's highlighted that SL is used for exploration, and this makes sense. It's relying on patterns and strategies learned from previous games, enabling the search to quickly identify potentially good moves without needing to evaluate every possibility from scratch.

  - Ole also writes that RL used for proposing new moves aren't as explorative. They tend to fixate on a few moves. Therefore SL tends to triumph for this.

- **RL (Reinforcement Learning)** value function, on the other hand, is used to score positions. Once moves are proposed using SL, the RL value function evaluates them. The value function is trained based on the outcomes of the games played, so it's learning from the environment and can adjust its evaluations over time.

###### Insights:

This combination of MCTS with deep RL is powerful because it combines the best of both worlds: the tree search explores possible game scenarios extensively, while the deep neural networks (trained using both SL and RL) provide strong evaluations of game positions and suggest promising moves. 



## Policy gradients

For the Go case, we only get one reward, and that is given when the game terminates.

The slide discusses a policy network and how it interacts with roll-outs in the context of reinforcement learning.

**The policy network**, $$\quad$$ denoted by $$p_\theta(a|s)$$ , gives a distribution over possible actions $$a$$ given a state $$s$$. Essentially, when your agent observes a state, the policy network tells it the likelihood of taking each possible action.

**Sampling roll-outs**, $$\quad$$ A roll-out in the context of reinforcement learning, refers to the process of simulating forward in time from a given state using the current policy. It's like saying, "If I take this action now, and then follow my policy thereafter, what sequence of actions am I likely to take?" 
The equation given for $$\textbf{a}$$  is demonstrating this: over a sequence of T time steps, the actions are being sampled from the policy:

$$
\textbf{a} = a_1, \dots , a_T \sim p_\theta(\textbf{a}|\textbf{s})
$$

**Joint probability** $$\quad$$ The equation 3 represents the joint probability of a sequence of actions, given a sequence of states. Simply, it multiplies the porbabilities of taking each action at each time step, given the previous state.

$$
p_\theta(\textbf{a}|\textbf{s}) = \prod_{t=1}^Tp_\theta (a_t|s_{t-1})
$$

**Expected Discounted Cumulative Reward** $$\quad$$ This is the meat of reinforcement learning. It's the expected sum of rewards you'd receive if you follow a policy starting from a given state. The integral is essentially saying, "For all possible roll-outs (i.e., sequences of actions) you can take, multiply the probability of that roll-out by the reward of that roll-out, and then sum it all up." The "discounted" part typically means that future rewards are worth less than immediate ones, but this equation seems to exclude the discounting factor (commonly denoted by a gamma, as we saw earlier).

$$
\mathbb{E}[R|\theta] = \int_{\text{roll-outs} }{R(\textbf{a})p_\theta(\textbf{a}|\textbf{s})}d\textbf{a}
$$


**Taking the Gradient with Respect to θ** $$\quad$$ The objective in many reinforcement learning algorithms is to maximize the expected reward. If we take the gradient with respect to $$\theta$$, we can use:

$$
\grad_\theta p_\theta(\textbf{a}|\textbf{s}) = p_\theta(\textbf{a}|\textbf{s})\grad_\theta\log p_\theta(\textbf{a}|\textbf{s})
$$

The above identity is quite crucial and can be derived from the properties of the gradient and the logarithm. Without going too deep into the proof, this identity basically relates the gradient of the policy itself to the gradient of the logarithm of the policy. The logarithm helps because it can transform products into sums, which are often easier to handle, especially when dealing with probabilities. 

**Replacing Integral by Average over S Roll-outs** $$\quad$$ Instead of calculating the exact gradient of the expected reward, which might involve integrals over the entire state and action spaces, we can approximate this gradient using Monte Carlo methods. We do this by sampling trajectories (or roll-outs) from our current policy. For each trajectory, we can compute the total reward and weight this with the gradient of the log-probability of the trajectory under our policy.

By replacing the integral by an average over $$S$$ roll-outs:

$$
\grad_\theta \mathbb{E}[R|\theta]\approx \frac{1}{S}\sum_{s = 1}^S R(\textbf{a}^{(s)})\grad_{\theta}\log p_\theta(\textbf{a}^{(s)}|\textbf{s}^{(s)})
$$

This equation essentially says that to estimate the gradient of our expected reward, we'll sample $$S$$ trajectories and for each trajextory we compute the reward $$R(\textbf{a}^{(s)})$$ and multiply it with the gradient of the log probability of the action taken given the state under our current policy.

- The gradient estimate does have a high variance but we can correct for that.
- This is a known problem in policy gradient methods. The intuition is that since we're sampling trajectories, two different runs can produce very different results. There are techniques like using a baseline (often a value function) to reduce this variance, but that's a topic for another deep dive.


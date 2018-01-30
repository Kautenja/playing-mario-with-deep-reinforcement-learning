
# Abstract

*   Evolution Strategies is (basically) a gradient-based technique, just a
	stochastic one using evolution based _metaheuristics_.
*   GA are _non-gradient_ based.
*   they find GA outperforms ES, A3C, and DQN
*   **IMPORTANT** They're optimizing _weights_m **NOT** _architectures_.

> Deep artificial neural networks (DNNs) are typically trained via gradient-
based learning algorithms, namely backpropagation. Evolution strategies (ES)
can rival backprop-based algorithms such as Q-learning and policy gradients on
challenging deep reinforcement learning (RL) problems. However, ES can be
considered a gradient-based algorithm because it performs stochastic gradient
descent via an operation similar to a finite-difference approximation of the
gradient. That raises the question of whether non-gradient-based evolutionary
algorithms can work at DNN scales. Here we demonstrate they can: we evolve the
weights of a DNN with a simple, gradient-free, population-based genetic
algorithm (GA) and it performs well on hard deep RL problems, including Atari
and humanoid locomotion. The Deep GA successfully evolves networks with over
four million free parameters, the largest neural networks ever evolved with a
traditional evolutionary algorithm. These results (1) expand our sense of the
scale at which GAs can operate, (2) suggest intriguingly that in some cases
following the gradient is not the best choice for optimizing performance, and
(3) make immediately available the multitude of techniques that have been
developed in the neuroevolution community to improve performance on RL
problems. To demonstrate the latter, we show that combining DNNs with novelty
search, which was designed to encourage exploration on tasks with deceptive or
sparse reward functions, can solve a high-dimensional problem on which reward-
maximizing algorithms (e.g. DQN, A3C, ES, and the GA) fail. Additionally, the
Deep GA parallelizes better than ES, A3C, and DQN, and enables a state-of-the-
art compact encoding technique that can represent million-parameter DNNs in
thousands of bytes.

# Conclusions

*   other old GA, ES, etc. are open to experimentation. Let the races begin!

> Our work introduces a Deep GA, which involves a simple parallelization trick
that allows us to train deep neural networks with GAs. We then document that
GAs are surprisingly competitive with popular algorithms for deep
reinforcement learning problems, such as DQN, A3C, and ES, especially in the
challenging Atari domain. We also showed that interesting algorithms developed
in the neuroevolution community can now immediately be tested with deep neural
networks, by showing that a Deep GApowered novelty search can solve a
deceptive Atari-scale game. It will be interesting to see future research
investigate the potential and limits of GAs, especially when combined with
other techniques known to improve GA performance. More generally, our results
continue the story – started by backprop and extended with ES – that old,
simple algorithms plus modern amounts of computation can perform amazingly
well. That raises the question of what other old algorithms should be
revisited.

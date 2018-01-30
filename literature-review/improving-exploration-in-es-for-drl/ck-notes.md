
# Abstract

*   ES can train DNN about as well as Q-learning and policy gradient methods
*   Hybridizes _novelty search_ and _quality diversity_ algorithms with ES
	can improve performance on _sparse_ or _deceptive_ RL tasks
	*   retain scalability
	*   avoid local optima in ES
*   Atari and Humanoid Locomotion

> Evolution strategies (ES) are a family of blackbox optimization algorithms
able to train deep neural networks roughly as well as Q-learning and policy
gradient methods on challenging deep reinforcement learning (RL) problems, but
are much faster (e.g. hours vs. days) because they parallelize better.
However, many RL problems require directed exploration because they have
reward functions that are sparse or deceptive (i.e. contain local optima), and
it is not known how to encourage such exploration with ES. Here we show that
algorithms that have been invented to promote directed exploration in small-
scale evolved neural networks via populations of exploring agents,
specifically novelty search (NS) and quality diversity (QD) algorithms, can be
hybridized with ES to improve its performance on sparse or deceptive deep RL
tasks, while retaining scalability. Our experiments confirm that the resultant
new algorithms, NS-ES and a version of QD we call NSR-ES, avoid local optima
encountered by ES to achieve higher performance on tasks ranging from playing
Atari to simulated robots learning to walk around a deceptive trap. This paper
thus introduces a family of fast, scalable algorithms for reinforcement
learning that are capable of directed exploration. It also adds this new
family of exploration algorithms to the RL toolbox and raises the interesting
possibility that analogous algorithms with multiple simultaneous paths of
exploration might also combine well with existing RL algorithms outside ES.

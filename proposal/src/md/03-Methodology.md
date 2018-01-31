# Methodology

\cite{deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative}
show that a _simple genetic algorithm_ (SGA) using pure Gaussian mutation with
no crossover can optimize the 4M parameters of a deep reinforcement agent
(originally introduced by \cite{human-level-control-through-deep-rl}) with
state-of-the-art performance. To further explore the potential of evolutionary
computation in this field, we plan to improve upon this algorithm by evolving
not only a weight-set $\theta$, but also the network _topology_, _components_,
and _hyperparameters_. This idea derives from the work of
\cite{evolving-deep-neural-networks}, who show that such algorithms produce
models that compete with the best man-made models. Ultimately, we aim to
aggregate these methods to produce a simple genetic algorithm that optimizes
the architecture and $\theta$ of a reinforcement agent's value predicting DNN.

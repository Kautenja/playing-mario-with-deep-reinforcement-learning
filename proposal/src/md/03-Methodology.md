# Methodology

\cite{deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative}
show that a _simple genetic algorithm_ (SGA) using pure Gaussian mutation with
no crossover can optimize the 4M parameters of a deep reinforcement agent
(originally introduced by \cite{human-level-control-through-deep-rl}) with
state-of-the-art performance. Although pure Gaussian mutation proves an
effective method, the introduction of a crossover operator could allow the
population to converge faster. Bearing such realization in mind, we will
investigate the possibility and effect of using four distinct crossover
operators with and without mutation: _BLX-0.0 crossover_,
_BLX-$\alpha$ crossover_, _single point crossover_, and _midpoint crossover_.
This list is by no means exhaustive, but allows a benchmark on a variety of
crossover techniques.

To further explore the potential of evolutionary computation in this field, we
plan to improve upon this GA by evolving not only a weight-set $\theta$, but
also the network _topology_, _components_, and _hyperparameters_. This idea
derives from the work of \cite{evolving-deep-neural-networks}, who show that
such algorithms produce models that compete with the best man-made models.
Ultimately, we aim to aggregate these methods to produce an SGA that optimizes
the architecture and $\theta$ of a reinforcement agent's value predicting DNN.
Again, we note this addition to be quite challenging, but remain open to the
possibility should implementing crossover operators not take the duration of
the class.

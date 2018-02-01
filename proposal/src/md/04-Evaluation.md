# Evaluation

We will benchmark our algorithm across a the full series of Atari 2600
games matching the work of
\cite{deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative,
human-level-control-through-deep-rl}. Should the benchmarking task prove
highly time consuming due to lack of available computational power, we will
select a subset of $\approx$ 5 games. The cumulative reward - final score of
the game - after $n$ training episodes provides an objective measure of the
performance of the algorithm. We will compare our results against those of
contemporary papers and implement a set of simpler deep RL agents for
comparative metrics internal to our study: Deep Q-Network (DQN), Double Deep
Q-Network (DDQN), Asynchronous Advantage Actor Critic (A3C), and random
search. This wide variety of sub-domains and techniques will validate or
disprove the ability of these new GAs to generalize across the Atari domain.

# Evaluation

<!--
\cite{open-ai-gym} provide an open source Python interface to popular RL
benchmarks including the suite of Atari games.
-->

We will benchmark our algorithm across a the full series of Atari 2600
game matching the work of
\cite{deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative,
human-level-control-through-deep-rl}. The cumulative reward - final score of
the game - after $n$ training episodes provides an objective measure of the
performance of the algorithm. To compare our results, we will implement
traditional strategies like Deep Q-Network (DQN), Double Deep Q-Network
(DDQN), Asynchronous Advantage Actor Critic (A3C), and random search. This
wide variety of sub-domains and techniques will validate or disprove the
ability of this new GA to generalize across different domains.

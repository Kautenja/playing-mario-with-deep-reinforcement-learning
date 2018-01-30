# Problem Statement

Recent work by researchers at Uber Labs shows the efficacy of evolutionary
algorithms in optimizing weights for _deep neural networks_ (DNN) on
image-based _reinforcement learning_ (RL) tasks. They find that the
introduction of _novelty search_ (NS) and _quality detection_ (QD) allows
their evolutionary agents to leverage exploration against raw fitness.
Although they achieve state-of-the-art performance on multiple RL benchmarks,
their works raises new questions about the potential of evolutionary
algorithms in deep RL. This project aims to explore the new problem space by
investigating the possibility of evolving not only network weights, but also
_topologies_, _components_, and _hyperparameters_. In doing so we hope to
achieve or surpass state of the art performance on the Atari benchmark with a
model designed, optimized, and tuned entirely automatically.

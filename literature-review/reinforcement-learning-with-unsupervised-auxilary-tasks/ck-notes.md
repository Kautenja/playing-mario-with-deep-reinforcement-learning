
# Abstract

*   introduce an agent that maximizes more than cumulative reward
*   novel mechanism for focusing this on _extrinsic_ reward so learning can
	adapt to most relevant aspects of the task.
*   outperforms state of the art on
	*   Atari (880% human performance)
	*   3D FPS Labrynth (87% human performance)

> Deep reinforcement learning agents have achieved state-of-the-art results by
directly maximizing cumulative reward. However, environments contain a much
wider variety of possible training signals. In this paper, we introduce an
agent that also maximizes many other pseudo-reward functions simultaneously by
reinforcement learning. All of these tasks share a common representation that,
like unsupervised learning, continues to develop in the absence of extrinsic
rewards. We also introduce a novel mechanism for focusing this representation
upon extrinsic rewards, so that learning can rapidly adapt to the most
relevant aspects of the actual task. Our agent significantly outperforms the
previous state-of-the-art on Atari, averaging 880% expert human performance,
and a challenging suite of first-person, three-dimensional Labyrinth tasks
leading to a mean speedup in learning of 10× and averaging 87% expert human
performance on Labyrinth.

# Introduction & Motivations

Driven by contemporary advances in reinforcement learning and a passion for
game playing, we explore the potential of using a deep reinforcement agent to
play the classic video game Super Mario Bros. We apply cutting edge updates
to the agent's model to attempt to produce state-of-the art performance on
this new benchmark. We also assess the performance of the agent using 5 Atari
games from the standard benchmark to compare performance of our agent with
the current state-of-the-art on a familiar task.

<!-- TODO: present some results -->

We initially proposed exploring the potential of genetic algorithms in
evolving the agent opposed to the standard gradient-based approach. We
planned to implement cutting edge techniques like Deep-$Q$ networks, A3C, and
Deep-SARSA as a baseline. With no original experience with reinforcement
learning, we spent a great deal of time understanding, debugging, and
implementing Deep-$Q$ as a baseline. Having underestimated the hardware
requirements of genetic optimization on this task, we simply ran out of time.

# Conclusion

In this study we explored the generality of Dueling \ac{DDQN} by attempting to
train it on a more complicated benchmark than Atari 2600, Super Mario Bros.
Although we encountered engineering problems that slowed the wall clock
performance of our agent, we demonstrate that the Dueling \ac{DDQN} does in
fact adapt well to this task with no hyperparameter tuning (other than what
is necessary to accommodate the hardware).

We note a future research direction in the emulator that provides the ground
work for this task. The client-server pattern that supports the current
infrastructure for the emulator introduces massive overhead as well as
instability issues that at worst result in the emulator being randomly killed
by the Linux kernel.

Additionally, we suggest experimentation with alternative reward spaces
$\mathcal{R}$ for the task. An ineffective reward scheme can obfuscate the
task making the agent learn more slowly. For instance, rewarding the \ac{SMB}
agent for killing enemies as well as penalizing it for dying (when it touches
enemies) could help \ac{DQN} more quickly learn to extract enemies from the
pixel space.

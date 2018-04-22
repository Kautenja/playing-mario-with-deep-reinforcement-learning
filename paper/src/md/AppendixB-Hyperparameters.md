\clearpage

# Hyperparameters

| Parameter        | Atari        | \ac{NES}   | Description                                          |
|:-----------------|:-------------|:-----------|:-----------------------------------------------------|
| $k$              | $4$          | $1$        | Frame skip                                           |
| $l$              | $4$          | $4$        | Agent history                                        |
| $N$              | $1e6$        | $5e5$      | Replay memory size                                   |
| $n$              | $32$         | $32$       | Replay memory batch size                             |
| $\alpha$         | $5e$-$2$     | $5e$-$2$   | Learning rate of Adam                                |
| Adam $\beta_1$   | $0.9$        | $0.9$      | Weight of first moment of Adam                       |
| Adam $\beta_2$   | $0.99$       | $0.99$     | Weight of the second moment of Adam                  |
| $\gamma$         | $0.99$       | $0.99$     | Factor for discounting future rewards                |
| $m$              | $4$          | $4$        | The number of actions before a replay                |
| $\epsilon_0$     | $1.0$        | $1.0$      | The starting value for exploration rate              |
| $\epsilon_f$     | $0.1$        | $0.1$      | The final value of the exploration rate              |
| $E$              | $1e6$        | $1e6$      | The number of frames to decay the exploration rate   |
| $\epsilon_{val}$ | $0.05$       | $0.05$     | The constant exploration rate for validation         |
| $t$              | $1e4$        | $1e4$      | The number of frames between target network updates  |
| $p$              | $5e4$        | $5e4$      | The number of random initial observations            |
| $T$              | $1e7$        | $<< 1e7$   | The total number of training frames                  |
| $v$              | $100$        | $100$      | The number of games in a validation run              |

Table: The hyperparameter configuration held constant in this experiment.
Because of instability in the \ac{NES} emulator, we had to train over multiple
different sessions of learning, and thus have no way of knowing how many
frames it saw. We can guarantee that it is $<< 1e7$.
\label{tab:hyperparameters}

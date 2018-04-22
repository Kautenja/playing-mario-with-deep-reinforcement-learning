\clearpage

# Hyperparameters

| Parameter        | Atari        | \ac{NES}   |
|:-----------------|:-------------|:-----------|
| $k$              | $4$          | $1$        |
| $l$              | $4$          | $4$        |
| $N$              | $1e6$        | $5e5$      |
| $n$              | $32$         | $32$       |
| $\alpha$         | $5e$-$2$     | $5e$-$2$   |
| Adam $\beta_1$   | $0.9$        | $0.9$      |
| Adam $\beta_2$   | $0.99$       | $0.99$     |
| $\gamma$         | $0.99$       | $0.99$     |
| $m$              | $4$          | $4$        |
| $\epsilon_0$     | $1.0$        | $1.0$      |
| $\epsilon_f$     | $0.1$        | $0.1$      |
| $\epsilon_{val}$ | $0.05$       | $0.05$     |
| $t$              | $1e4$        | $1e4$      |
| $p$              | $5e4$        | $5e4$      |
| $T$              | $1e7$        | $<< 1e7$   |
| $v$              | $100$        | $100$      |

Table: The hyperparameter configuration held constant in this experiment.
Because of instability in the \ac{NES} emulator, we had to train over multiple
different sessions of learning, and thus have no way of knowing how many
frames it saw. We can guarantee that it is $<< 1e7$.

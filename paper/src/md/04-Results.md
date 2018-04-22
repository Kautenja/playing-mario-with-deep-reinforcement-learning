# Results

We combine both \ac{DDQN} and Double Deep-$Q$ to produce the Double Dueling
Deep-$Q$ Network. We use the hyperparameters in Table
\ref{tab:hyperparameters}. Because of issues with the \ac{NES} emulator
crashing during training, we were unable to keep track of the number of
training frames $T$ and performance metrics such as the loss and reward of
each training episode. We can however, validate an untrained and trained
agent to confirm learning.

## Atari

Our agent performs modestly on the Atari benchmark. Training for $1e7$ frames
on the \cite{OhioSupercomputerCenter1987} consumed $\approx 15$ hours per
game and produced the results in Table \ref{tab:atari-results}. We mistakenly
recorded the _clipped_ reward value instead of the _in-game score_, resulting
in poor comparisons to the other Deep-RL literature. However, the results
confirm that the algorithm converges to a nearly optimal policy achieving
higher rewards than the initial state (randomness).

Table: Validation results of Double \ac{DDQN} before and after training to
play games for $1e7$ frames on Atari benchmarks. It is worth noting that these
values represent _clipped_ rewards, and as such do not compare well to the
previous works of others. \label{tab:atari-results}

+---------+------------+----------+----------+------------+-----------------+
|         |   Breakout |   Enduro |     Pong |   Seaquest |   SpaceInvaders |
+=========+============+==========+==========+============+=================+
| Final   |   85.4343  |  1184.06 |  19.6364 | 263.818    |         52.9899 |
+---------+------------+----------+----------+------------+-----------------+
| Initial |   -3.43434 |     0    | -21      |   0.585859 |          6.9899 |
+---------+------------+----------+----------+------------+-----------------+

### Pong

Fig. \ref{fig:train-Pong} describes the per-episode loss and reward
(cumulative) seen by the agent. The loss increases as the agent begins to
explore and locate the optimal policy. Once the policy is found, the loss
converges over $\approx 500$ episodes resulting in an agent with master
competency. Although this learning curve suggest that Double \ac{DDQN}
produces a clean learning signal, games with complicated reward schemes or
noisy inputs result in a far noisier output signal.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.95\textwidth]{img/games/Pong/training}
\caption{Per-episode training results for the Double \ac{DDQN} on the game
\textbf{Pong}. The agent converges to the optimal policy relatively quickly
around episode 1100.}
\label{fig:train-Pong}
\end{figure}

## Super Mario Bros.

TODO:

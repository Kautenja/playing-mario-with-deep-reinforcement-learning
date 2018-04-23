# Results

We combine both \ac{DDQN} and Double Deep-$Q$ to produce the Double Dueling
Deep-$Q$ Network. We use the hyperparameters in Table
\ref{tab:hyperparameters}. Because of issues with the \ac{NES} emulator
crashing during training, we were unable to keep track of the number of
training frames $T$ and performance metrics such as the loss and reward of
each training episode. We can, however, validate both an untrained agent
and a fully trained one to confirm learning.

## Atari

Our agent performs modestly on the Atari benchmark. Training for $T = 1e7$
frames on the \cite{OhioSupercomputerCenter1987} consumed $\approx 15$ hours
per game and produced the results in Table \ref{tab:atari-results}. In only
$\frac{1}{5}$ as many training frames, the Double \ac{DDQN} outperforms
vanilla \ac{DQN} on **Pong**, **Enduro**, and **Seaquest**. However, it fails
to compete on the **Breakout** and **SpaceInvaders** tasks.

Table: Mean score of \ac{DQN}, Double \ac{DDQN}, random agents on $v = 100$
episodes after training. It is worth noting that these values represent
in-game scores, not rewards seen by the agent.
\label{tab:atari-results}

|          |   Breakout |   Enduro |     Pong |   Seaquest |   Space Invaders |
|:---------|-----------:|---------:|---------:|-----------:|-----------------:|
| DDDQN    |        363 |  **330** | **19.7** |   **8015** |              999 |
| DQN      |    **401** |      301 |     18.9 |       5286 |         **1976** |
| Random   |          1 |    0     | -21      |        103 |              138 |

### Pong

Fig. \ref{fig:train-Pong} describes the per-episode loss and reward
(cumulative) seen by the agent. The loss increases as the agent begins to
explore and locate the optimal policy. Once the policy is found, the loss
converges over $\approx 500$ episodes resulting in an agent with master
competency. Although this learning curve suggests that Double \ac{DDQN}
produces a clean learning signal, games with complicated reward schemes or
noisy inputs result in a more erratic output signal.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.95\textwidth]{img/games/Pong/training}
\caption{Per-episode training results for the Double \ac{DDQN} on the game
\textbf{Pong}. The agent converges to the optimal policy relatively quickly
around episode 2000.}
\label{fig:train-Pong}
\end{figure}

## Super Mario Bros.

Table: Statistics from the Random and DDDQN agent respectively. The values
represent Mario's mean terminal $x$ position among $v = 100$ validation
games. \label{tab:smb-results}

|         |  Min | Mean |  Max |
|:--------|-----:|-----:|-----:|
| Random  |   17 |  153 |  336 |
| DDDQN   |   45 |   64 |  272 |

We were unable to collect data to visualize the learning process for Super
Mario Bros. on a per episode basis. However, we present the validation scores
of the agent compared to randomness in Table. \ref{tab:smb-results}. Although
the DDDQN outperforms the random agent, it still performs relatively poorly,
making it to the end of the level 0 times. Because Super Mario Bros is
arguably more complex than all of the Atari 2600 games, it's fair to assume
that it should need more training time or potentially a network with larger
capacity to better fit the space. We note the complexity of the reward scheme
in Super Mario Bros. as a potential future research direction. Without a
strong and effective reward signal, the agent cannot competently learn the
correct task. We simply reward the agent for moving right and penalize it for
moving left or dying. Future work can explore rewarding the agent for killing
enemies or collecting coins and penalizing the agent for losing time on the
clock.

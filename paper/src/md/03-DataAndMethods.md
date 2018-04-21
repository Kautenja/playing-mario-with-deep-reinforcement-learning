# Data & Methods

<!-- TODO: discuss games and tasks? -->
<!-- TODO: notation table -->
<!-- TODO: references -->

## Reinforcement Learning Model

The task of playing a game to maximize score closely models a _Markov
decision process (MDP)_ shown in Fig. \ref{fig:mdp}.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.6\textwidth]{img/mdp}
\caption{The general form of a Markov decision process (MDP). An agent
produces an action that actuates some environment. The environment produces a
new state, and a reward that the agent observes. The process repeats
indefinitely.}
\label{fig:mdp}
\end{figure}

## Preprocessing

Following the work of \cite{human-level-control-through-deep-rl}, we apply a
down-sampling $\phi$ to each frame produced by the emulator. $\phi$ first
crops the RGB image to the "playable" area of the screen. We parameterize
$\phi$ with pairs $(x_l, x_r)$ of the horizontal pixels to crop and
$(y_t, y_b)$ of the vertical pixels to crop from the RGB image. After
cropping $\phi$ down-samples the RGB image to a single B&B channel and resizes
the image to $84 \times 84$ using a bilinear interpolation. This down-sampling
step helps reduce dimensionality of states to make computation easier.

## Frame Skipping

<!-- TODO: update k value if we use mario instead -->

\cite{human-level-control-through-deep-rl} show that agents perform well
despite experiencing a fraction of their environment. We apply a $k$ frame
skip mechanism to allow the agent a linear speedup of $\leq k$. This
mechanism holds each action produced by the agent for $k$ frames, returning
the total reward over the $k$ frames. We use the point-wise maximum between
the last two frames as the next state to account for flickering sprites.
Intermediary frames are dropped, the agent never sees them. Matching
\cite{human-level-control-through-deep-rl}, we apply $k = 4$ to all tasks.

## Frame Stacking

To provide the agent an understanding of directionality and velocity of
sprites in games, \cite{human-level-control-through-deep-rl} stacks a history
of $l$ frames as the current state. It is worth noting that we only keep a
history of the frames that the agent reacts to in the stack; the $k - 1$
frames that are skipped with held actions are never seen by the agent. Fig.
\ref{fig:frame-skip} illustrates roughly how this process works for $l = 3$
and an arbitrary $k \geq 1$. This study uses a value $l = 4$ for all games.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.6\textwidth]{img/frame-skip}
\caption{The process for skipping and stacking frames. An agent reacts every
$k$ frames, holding an action during intermediary frames. The agent then keeps
the last $l$ frames it reacted to as the current state.}
\label{fig:frame-skip}
\end{figure}

## Experience Replay

<!-- TODO: update value of N if we use Mario. -->
<!-- TODO: check FCEUX spelling and reference -->
<!-- TODO: note that prioritized is better, ran out of time -->

To build a dataset for training an agent's reward estimator, we use an
experience replay queue. An agent in some state $s$ performs an action $a$ to
produce a new state $s'$, a reward $r$, and a flag denoting whether the
episode (game) has ended $d$. The agent stores the experience as the tuple
$e = (s, a, r, d, s')$ in a FIFO queue $D = {e_1, ..., e_N}$ of at most $N$
total experiences. To generate training data, the agent randomly draws a
mini-batch $d$ of $n$ experiences using a uniform distribution with
replacement (i.e. $d = \{e_1, ..., e_n\} \sim U(D)$).
\cite{human-level-control-through-deep-rl} apply $N = 1e6$ previous
experiences with mini-batchs of size $n = 32$. We match these values in our
Atari experiments, but use a size of $N = 5e5$ in the Mario experiment due to
hardware restrictions imposed by FCEUX, the NES emulator.

## Deep-Q Learning

Although traditional Q-learning is effective in some classical reinforcement
learning problems, the quality table suffers from the state complexity of
NP-Hard tasks such as playing games using pixel space.
\cite{human-level-control-through-deep-rl} presented the Deep-Q algorithm to
combat this weakness by confidently estimating the quality table using a Deep
Neural Network (DNN). Eqn. \ref{eqn:q-alg} shows the original formulation of
the Q-learning algorithm. The algorithm replaces the quality value for the
current state action pair $(s, a)$ with the sum of the current quality
estimate $Q(s, a)$ and the learning rate adjusted reward $r$ and estimated
future reward $Q(s', a')$ from the next state $s'$ over all possible actions
$a' \in A$.

<!-- TODO: too wordy? -->
<!-- TODO: is it worth talking about Bellman -->

<!-- This algorithm employs a greedy policy to always
select the action that maximizes future reward. It is worth noting that the
process of updating the $Q$ table from its own estimates is a process known
as bootstrapping. -->

\begin{equation}
Q(s, a) \gets Q(s, a) + \alpha \bigg(r + \gamma \max_{a' \in A}Q(s', a') - Q(s, a) \bigg)
\label{eqn:q-alg}
\end{equation}

The Deep-Q algorithm

<!-- \begin{equation}
Q(s', a') \gets (1 - \alpha)Q(s_t, a_t) + \alpha \bigg(r_t, + \gamma • \max_{a \in A}Q(s_{t+1}, a) \bigg)
\label{eqn:deep-q-alg}
\end{equation}
 -->

-   bootstrapping
-   bellman optimality
-   table
-   approximate table
-   NP completeness of problem

### Double Deep-Q Learning

update target network

### Dueling Deep-Q Learning

approximate state value and action value then define a novel layer for
aggregating them into Q values over all actions.

## Hardware Configuration

\cite{OhioSupercomputerCenter1987}

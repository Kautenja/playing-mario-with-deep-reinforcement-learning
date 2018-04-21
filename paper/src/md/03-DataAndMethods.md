# Data & Methods

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

Following the work of {DEEP-MIND}, we apply a down-sampling $\phi$ to each
frame produced by the emulator. $\phi$ first crops the RGB image to the
"playable" area of the screen.  We parameterize $\phi$ with pairs $(x_l, x_r)$
of the horizontal pixels to crop and $(y_t, y_b)$ of the vertical pixels to
crop from the RGB image. After cropping $\phi$ down-samples the RGB image to
a single B&B channel and resizes the image to $84 \times 84$ using a
bilinear interpolation. This down-sampling step helps reduce dimensionality of
states to make computation easier.

## Frame Skipping

{DEEP-MIND} show that agents perform well despite experiencing a fraction of
their environment. We apply a $k$ frame skip mechanism to allow the deep-q
agent a linear speedup of $\leq k$. This mechanism holds each action produced
by the agent for $k$ frames, returning the total reward over the $k$ frames.
We use the point-wise maximum between the last two frames as the next state to
account for flickering sprites. Intermediary frames are dropped, the agent
never sees them.

## Frame Stacking

To provide the agent an understanding of directionality and velocity of
sprites in games, {DEEP MIND} stacks a history of $l$ frames as the current
state. It is worth noting that we only keep a history of the frames that
the agent reacts to in the stack; the $k - 1$ frames that are skipped with
held actions are never seen by the agent. Fig. \ref{fig:frame-skip}
illustrates roughly how this process works for $l = 3$ and an arbitrary
$k \geq 1$.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.6\textwidth]{img/frame-skip}
\caption{The process for skipping and stacking frames. An agent reacts every
$k$ frames, holding an action during intermediary frames. The agent then keeps
the last $l$ frames it reacted to as the current state.}
\label{fig:frame-skip}
\end{figure}

## Experience Replay


## Q-Learning Algorithm

-   bootstrapping
-   bellman optimality
-   table

### Deep-Q Learning

-   approximate table
-   NP completeness of problem

\cite{OhioSupercomputerCenter1987}

### Double Deep-Q Learning

update target network

### Dueling Deep-Q Learning

approximate state value and action value then define a novel layer for
aggregating them into Q values over all actions.

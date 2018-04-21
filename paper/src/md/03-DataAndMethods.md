<!--
    NP-something SMB
http://erikdemaine.org/papers/Mario_FUN2016/paper.pdf
 -->

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
the last $l$ frames it reacted to as the current state. This figure implies
an arbitrary $k$ value, but an $l$ value of $3$.}
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

\begin{equation}
Q(s, a) \gets Q(s, a) + \alpha \bigg(r + \gamma \max_{a' \in A}Q(s', a') - Q(s, a) \bigg)
\label{eqn:q-alg}
\end{equation}

Deep-Q approximates the $Q$ table using a neural network and updates states
by back-propagating the error as a result of the loss function shown in
Eqn. \ref{eqn:deep-q-alg}. We define the expected label $y = r + Q(s', a')$ as
the expected future reward, and the predicted label $\hat{y} = Q(s, a)$ as
the estimated future reward for the current state action pair.

\begin{equation}
L_i(\theta_i) =
\mathbb{E}_{e' \sim U(D)} \bigg[
\bigg( r + \gamma \max_{a' \in A} Q(s', a', \theta) - Q(s, a, \theta) \bigg)^2
\bigg]
\label{eqn:deep-q-alg}
\end{equation}

\begin{equation}
L_{\delta}(y, \hat{y}) = \begin{cases}
      \frac{1}{2} (y - \hat{y})^2                & |y - \hat{y}| \leq \delta \\
      \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \textbf{otherwise} \\
\end{cases}
\end{equation}

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

<!--
## Hardware Configuration

We use two distinct hardware configurations in our experiment. For the Atari
range of experiments, we use the servers of \cite{OhioSupercomputerCenter1987}.

TODO: get specs of the servers

The \cite{OhioSupercomputerCenter1987} provides no super user access, necessary
to install the NES emulator, FCEUX, used in the Mario experiment. We instead
run this experiment locally on a workstation with a 4.2GHz Intel Core i5,
nVidia GTX1070, and 32GB of 3200MHz RAM. Unlike the Atari emulator, which is
written in Python, FCEUX is a standalone application that supports plugins
written in Lua. To interface with our Python stack, the use of a client server
pattern between game engine process and the agent process. Unfortunately, this
overhead drastically impedes the agent's ability to interact with the
environment. We note a slowdown of $\approx 8$x as compared to the Atari
emulator based on agent frame rate. -->

# Data & Methods

## Reinforcement Learning Model

The task of playing a video game like those on the Atari 2600 or the \ac{NES}
closely models a _\ac{MDP}_ shown in Fig. \ref{fig:mdp}. In this case, an
agent experiences a game in pixel space $\mathcal{S}$ to produce an action
from a discrete action space $\mathcal{A}$ corresponding to a combination of
buttons on a controller. The environment responds to the action with a new
state from $\mathcal{S}$ and a reward from reward space $\mathcal{R}$. Games
define different reward spaces $\mathcal{R}$, but typically games grant
positive rewards for scoring points, moving forward in a level, or killing an
enemy; they grant negative rewards for losing points, backtracking in a
level, or dying.

\begin{figure}[!ht]
\centering
\includegraphics[width=0.6\textwidth]{img/mdp}
\caption{The general form of a \ac{MDP}. An agent in state
$s \in \mathcal{S}$ produces an action $a \in \mathcal{A}$ that actuates some
environment. The environment produces a new state $s' \in \mathcal{S}$, and a
reward $r \in \mathcal{R}$ that the agent observes. The process repeats
indefinitely.}
\label{fig:mdp}
\end{figure}

## Game Environment

We explore two distinct game environments in this study: the Atari 2600 and
the \ac{NES}. From the Atari 2600 we select five games: _Enduro_, _Breakout_,
_Pong_, _Seaquest_, and _Space Invaders_. From the \ac{NES} we select one
game, \ac{SMB}, due to hardware and time restrictions. We interface with both
environments using Open.ai Gym to allow agents to easily generalize across
the different environments with no alteration. We note that the emulator for
running \ac{NES} games, FCEUX, is up to $8x$ slower than the Atari emulator
due to a poorly designed client-server pattern that enables he Python stack
to communicate with the Lua based emulator.

## Preprocessing

Following the work of \cite{human-level-control-through-deep-rl}, we apply a
down-sampling function $\phi(s)$ to each frame produced by the emulator.
$\phi(s)$ first crops the RGB image to the playable area of the screen. We
parameterize $\phi(s)$ with pairs $x_{\phi} = (x_{left}, x_{right})$ of the
horizontal pixels to crop and $y_{\phi} = (y_{top}, y_{bottom})$ of the
vertical pixels to crop. After cropping, $\phi(s)$ down-samples the RGB image
to a single black & white channel (Y) and resizes the image to $84 \times 84$
pixels using bilinear interpolation. $\phi(s)$ reduces dimensionality of
states to better utilize hardware.

## Frame Skipping

\cite{human-level-control-through-deep-rl} show that agents perform well
despite experiencing a fraction of their environment. We apply a $k$ frame
skip mechanism to allow the agent a linear speedup of $\leq k$. This
mechanism holds each action produced by the agent for $k$ frames, returning
the total reward over the $k$ frames. We use the point-wise maximum between
the last two frames as the next state to account for flickering sprites.
Intermediary frames are dropped, the agent never sees them. Matching
\cite{human-level-control-through-deep-rl}, we apply $k = 4$ to all Atari
tasks. However, we get better performance on \ac{NES} tasks using $k = 1$ due
to locking from the FCEUX client-server pattern.

## Frame Stacking

To provide the agent an understanding of directionality and velocity of
sprites in games, \cite{human-level-control-through-deep-rl} stack a history
of $l$ frames as the current state. It is worth noting that we only keep a
history of the frames that the agent reacts to in the stack; the $k - 1$
frames that are skipped with held actions are never seen by the agent. Fig.
\ref{fig:frame-skip} illustrates roughly how this process works for $l = 3$
and an arbitrary $k \geq 1$. This study uses a value $l = 4$ for all games.

<!-- TODO: what about the max? -->

\begin{figure}[!ht]
\centering
\includegraphics[width=0.6\textwidth]{img/frame-skip}
\caption{The process for skipping and stacking frames. An agent reacts every
$k$ frames by holding an action during intermediary frames. The agent then
keeps the last $l$ frames it reacted to as the current state $s_i$. This
figure implies an arbitrary $k$ value, but an $l$ value of $3$.}
\label{fig:frame-skip}
\end{figure}

## Reward Clipping

\cite{human-level-control-through-deep-rl} found that clipping the rewards
at each step allows agent hyperparameters to generalize across a broad range
of reward spaces. They so by forcing each unique reward space $\mathcal{R}$
to the same reward space $\mathcal{R}' = \{-1, 0, 1\}$. This is as simple as
replacing a reward $r = sgn(r)$. We apply the same reward clipping in our
Atari and Super Mario Bros experiments.

## Experience Replay

To build a dataset for training an agent's reward estimator, we use an
experience replay queue. An agent in some state $s \in \mathcal{S}$ performs
an action $a \in \mathcal{A}$ to produce a new state $s' \in \mathcal{S}$, a
reward $r \in \{-1, 0, 1\}$, and a flag denoting whether the episode (game)
has ended $d \in \{0, 1\}$. The agent stores the experience as a tuple
$e = (s, a, r, d, s')$ in a FIFO queue $D = \{e_1, ..., e_N\}$ of at most $N$
total experiences. To generate training data, the agent randomly draws a
mini-batch $D'$ of $n$ experiences using a uniform distribution with
replacement (i.e. $D' = \{e_1, ..., e_n\} \sim U(D)$).
\cite{human-level-control-through-deep-rl} apply $N = 1e6$ previous
experiences with mini-batches of size $n = 32$. We match these values in our
Atari experiments, but use a replay queue size of $N = 5e5$ in the Mario
experiment due to hardware restrictions imposed by running on a local
workstation. Although \cite{human-level-control-through-deep-rl} report the
best results using RMSprop to optimize loss, we apply Adam in all of our
experiments with values $\alpha = 2e$-$5$, $\beta_1 = 0.9$, and
$\beta_2 = 0.99$.

## Deep-$Q$ Learning

Although traditional $Q$-learning excels in some classical reinforcement
learning problems, the quality table suffers from the state complexity of
NP-Hard tasks such as playing games using pixel space.
\cite{human-level-control-through-deep-rl} presented the \ac{DQN} to
combat this weakness by confidently estimating the quality table using a
\ac{DNN}. Eqn. \ref{eqn:q-alg} shows the original formulation of the
Q-learning algorithm. The algorithm replaces the quality value for the
current state-action pair $(s, a)$ with the sum of the current quality
estimate $Q(s, a)$ and the learning rate adjusted reward $r$ and discounted
estimated future reward $\gamma Q(s', a')$ from the next state $s'$ over all
possible actions $a' \in \mathcal{A}$.

\begin{equation}
Q(s, a) \gets
Q(s, a) +
\alpha \bigg(
r + \gamma \max_{a' \in \mathcal{A}}Q(s', a') - Q(s, a)
\bigg)
\label{eqn:q-alg}
\end{equation}

Deep-Q approximates the $Q$ table using a \ac{DNN} bearing weights $\theta$.
And, it updates by back-propagating error from mini-batches of replay data
$D' \sim U(D)$. For each sample in an arbitrary mini-batch $D'$, we define a
ground truth target $y$ (Eqn. \ref{eqn:deep-q-y}) and predicted value
$\hat{y}$ (Eqn. \ref{eqn:deep-q-y-hat}). We use the same \ac{DQN} presented
by \cite{human-level-control-through-deep-rl} shown in Fig. \ref{fig:dqn}.

\begin{equation}
y = r + (1 - d) \gamma \max_{a' \in \mathcal{A}} Q(s', a', \theta)
\label{eqn:deep-q-y}
\end{equation}

\begin{equation}
\hat{y} = Q(s, a, \theta)
\label{eqn:deep-q-y-hat}
\end{equation}

With definitions of target values $y$ and estimated values $\hat{y}$, we
define the loss function in Eqn. \ref{eqn:deep-q-alg}. Following
\cite{human-level-control-through-deep-rl}, we clip the gradient within the
continuous range $[-1, 1]$ using _Huber Loss (Eqn. \ref{eqn:huber})_ with
$\delta = 1$.

\begin{equation}
L(\theta) =
\mathbb{E}_{(s, a, r, d, s') \sim U(D)} \bigg[ L_{\delta}(y, \hat{y}) \bigg]
\label{eqn:deep-q-alg}
\end{equation}

\begin{equation}
L_{\delta}(y, \hat{y}) = \begin{cases}
      \frac{1}{2} (y - \hat{y})^2                & |y - \hat{y}| \leq \delta \\
      \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \textbf{otherwise} \\
\end{cases}
\label{eqn:huber}
\end{equation}

The algorithm starts by performing $p = 5e4$ random steps to fill the replay
memory with random data. The agent then experiences however many episodes it
can in $T$ total frames. The agent replays a memory every $m = 4$ actions. We
take the average score of $v = 100$ consecutive validation games before and
after to show the change in distribution of collected rewards. And, we
collect the reward and loss per episode to visualize convergence of the
algorithm to a nearly optimal policy.

#### $\epsilon$-greedy

When predicting actions in training and validation, the agent uses an
$\epsilon$-greedy policy to encourage exploration. That is to say, with some
probability $\epsilon$, an agent produces a random action instead of the
greedy action from the \ac{DQN}. $\epsilon$ decays over time to allow the
agent to take slow control of the environment.
\cite{human-level-control-through-deep-rl} decay $\epsilon$ over $1e6$ states
from $1.0$ to $0.1$ using a _linear_ schedule. For validation, they employ a
static $\epsilon = 0.05$. We use the same values, but decay $\epsilon$ using
a _geometric_ schedule.

### Double Deep-Q Learning

\cite{human-level-control-through-deep-rl} show that updates to the same
\ac{DQN} that defines targets results in unstable learning. A simple
resolution to this problem, Double Deep-$Q$ Learning, introduces an identical
model $\theta_{target}$ for determining the ground truth targets shown in
Eqn. \ref{eqn:double-deep-q-y}. Back-propagation continues to update $\theta$
which replaces $\theta_{target}$ every $t$ experiences. Our experiments apply
a standard $t = 1e4$.

\begin{equation}
y = r + (1 - d) \gamma \max_{a' \in \mathcal{A}} Q(s', a', \theta_{target})
\label{eqn:double-deep-q-y}
\end{equation}

### Dueling Deep-Q Learning

\cite{dueling-deep-q} presented an additional improvement to the \ac{DQN},
the Dueling Deep-Q Network, which replaces the densely connected network with
a new model that separates the data into two distinct streams, one that
estimates a scalar state value $V(s, \theta)$, and another that estimates
action advantage values $A(s, a, \theta)$. It combines these independent
streams into the value $Q(s, a, \theta)$ using the computational layer
described in Eqn. \ref{eqn:dueling-deep-q}. Fig. \ref{fig:dueling-dqn}
provides a graphical representation of this architecture.
\cite{dueling-deep-q} showed that the Dueling Deep-Q Network performs better
than the standard Deep-Q Architecture when the action space was large. We
anticipate this improvement to help in the Mario environment which features
$|\mathcal{A} = 14|$.

\begin{equation}
Q(s, a, \theta) = V(s, \theta) +
\bigg(
A(s, a, \theta) -
\frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a', \theta)
\bigg)
\label{eqn:dueling-deep-q}
\end{equation}

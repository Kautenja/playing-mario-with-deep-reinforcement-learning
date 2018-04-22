# Related Works

### Deep Reinforcement Learning

\cite{play-atari-with-deep-rl} conducted the first study of game playing with
deep reinforcement learning on the Atari 2600 platform. They presented a
novel method for approximating the $Q$ table in the $Q$ reinforcement
learning algorithm using a \ac{DNN} labeled the \ac{DQN}. Such an estimator
allowed them to estimate $Q$-values in large spaces like single-channel pixel
space – black & white images of a given width $w$ and height $h$ – using
constant space complexity $\Theta(|\theta|)$ opposed to the
$\Theta((w \times h)^{255})$ imposed by the standard $Q$ table. Their method,
_Deep-Q Learning_ outperformed human competency on game like **Breakout**,
**Pong**, and **Enduro**; but failed to do so on games like
**Space Invaders**, and **Seaquest**.

\cite{human-level-control-through-deep-rl} published an updated version of
the original work from \cite{play-atari-with-deep-rl} featuring an
improvement to the \ac{DQN}. They introduce a second \ac{DQN} for
establishing target values in the $Q$-learning action replay phase. In doing
so, the agent follows a more stable learning pattern, overestimating the
value of fewer states. \cite{double-q-learning} studied this
_Double Deep-$Q$ Learning_ method to validate its efficacy. They showed that
the tweak allowed the algorithm to outperform the original on each of:
**Breakout** **Pong**, **Enduro**, **Space Invaders**, and **Seaquest**.

\cite{dueling-deep-q} defined a new \ac{DQN} model, the _\ac{DDQN}_. The
novelty of this model lies in its inherent ability to estimate a scalar value
for each state, as well as measure the advantage of a potential action. Using
a simple computational layer, the network combines these estimates to produce
the same $Q$ values as the original \ac{DQN}. As such, the \ac{DDQN} is a
modular replacement requiring no actual change to the $Q$ learning algorithm.
\cite{dueling-deep-q} showed that the \ac{DDQN} outperforms the standard
\ac{DQN} for games with a lot of actions to choose from. They revealed
that this new model achieves better performance then the vanilla \ac{DQN} on
each of: **Breakout** **Pong**, **Enduro**, **Space Invaders**, and
**Seaquest**.

\cite{deep-neuroevolution-genetic-algorithms-are-a-competitive-alternative}
demonstrated that a distributed genetic optimizer will converge on more
optimal $\theta$ for a \ac{DQN} than the standard gradient-based approach on
most Atari 2600 tasks. By decoupling the GPU dependency and distributing the
search for $\theta$, they found state of the art agents in as few as $10$ minutes.
Interestingly, they showed that random search outperformed the gradient-based
approach in a few cases. They noted the ability of their agent to learn
complex games like **Seaquest** that Deep-$Q$ methods struggle with.

## Engineering Support

Modeling a video game as a reinforcement learning task embodies a wealth of
complicated engineering problems. \cite{ale1} approached the issue of
integrating an Atari 2600 emulator with Python using low-level C++ code in a
framework called the _\ac{ALE}_. \cite{ale2} later refined \ac{ALE} by
introducing new features like probabilistic frame skipping.
\cite{open-ai-gym} popularized the use of \ac{ALE} with their intuitively
simple, but powerful API for designing reinforcement agents in Python entitled
_Open.ai Gym_. \cite{openai-baselines} used Open.ai Gym as a framework to
recreate baselines of reinforcement learning papers. They provided extensions
to the default environments in Open.ai Gym to model environments used in
research papers. \cite{ppaquette} extended Open.ai Gym with an emulator for
playing \ac{NES} games. Namely, this framework provides an interface to the
classic \ac{NES} game, \ac{SMB}. It interacts with an instance of the
\ac{NES} emulator, \cite{fceux}, via a client-server pattern. It is worth
noting that this design pattern imposes a noticeable slowdown in performance
compared to games in \ac{ALE}.

### Deep Learning Infrastructure

\cite{keras} designed Keras, a high level interface to deep learning back-ends
in Python. This framework enables the fast construction of computational
graphs. And, it generalizes across popular back-ends including Theano,
TensorFlow, and CNTK. Our study applies the TensorFlow back-end.
\cite{OhioSupercomputerCenter1987} constructed an array of $160$ GPU nodes,
each bearing an nVidia P100 GPU, 28 core Intel Xeon CPU, and 132GB of RAM.
Our study runs experiments involving Atari 2600 on these machines, but not t
hose using \ac{NES} due to complications with FCEUX.

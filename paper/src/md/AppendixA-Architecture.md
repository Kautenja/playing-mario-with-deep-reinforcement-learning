\clearpage

<!-- TODO: show input size of (84 x 84 x l) tensor frames. -->
<!-- TODO: ReLu in diagrams? -->

# Architectures

\begin{figure}[!htp]
\centering
\includegraphics[width=0.8\textwidth]{img/dqn}
\caption{The Deep-Q Network (DQN) presented by
\cite{human-level-control-through-deep-rl}. A stack of $l$ frames passes
through 3 convolutional layers to extract features from pixel space. The
extracted features are flattened and passed to a densely connected network to
map the pixel state to a discrete action space $\mathcal{A}$. The output of
the network is a $|\mathcal{A}|$-dimensional vector of estimated future
rewards where the index $i$ in the vector represents the reward from taking
action $a_i$. The network features a ReLu activation between each layer and
a linear activation at the output layer. The network has $\approx 4e6$
weights.}
\label{fig:dqn}
\end{figure}

\begin{figure}[!htp]
\centering
\includegraphics[width=0.8\textwidth]{img/dueling-dqn}
\caption{The Dueling Deep-Q Network (DQN) presented by \cite{dueling-deep-q}.
The densely connected model separates the data into two streams that estimate
state value $V(s, \theta)$ (top) and action advantage $A(s, a, \theta)$
(bottom) respectively. The streams combine using a computational layer,
described in Eqn. \ref{eqn:dueling-deep-q}, to produce output values
$Q(s, a, \theta)$. Much like the original Deep-Q architecture, each layer is
separated by a ReLu activations; the output layer features a linear
activation. This network has $\approx 8e6$ weights (2x as many as the original
architecture).}
\label{fig:dueling-dqn}
\end{figure}

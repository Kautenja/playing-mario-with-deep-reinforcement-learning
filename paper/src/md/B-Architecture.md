\clearpage

<!-- TODO: show input size of (84 x 84 x l) tensor frames. -->
<!-- TODO: better diagrams with tikz -->
<!-- TODO: ReLu in diagrams and captions -->

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
action $a_i$. This network has $\approx 4e6$ weights.}
\label{fig:dqn}
\end{figure}

<!-- TODO: describe V, A, Q in diagram and in the caption -->

\begin{figure}[!htp]
\centering
\includegraphics[width=0.8\textwidth]{img/dueling-dqn}
\caption{The Dueling Deep-Q Network (DQN) presented by \cite{dueling-deep-q}.
The densely connected model separates the data into two streams that estimate
state value and action advantage respectively.
This network has $\approx 8e6$ weights.}
\label{fig:dueling-dqn}
\end{figure}

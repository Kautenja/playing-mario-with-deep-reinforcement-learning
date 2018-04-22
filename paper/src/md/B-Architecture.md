\clearpage

# Architectures

\begin{figure}[!htp]
\centering
\includegraphics[width=0.8\textwidth]{img/dqn}
\caption{The Deep-Q Network (DQN) presented by
\cite{human-level-control-through-deep-rl}. A stack of $l$ frames passes
through 3 convolutional layers to extract features from pixel space. The
extracted features are flattened and passed to a densely connected network to
map the pixel state to a discrete action $a \in A$. The output of the network
is a vector of estimated rewards where the index $i$ in the vector represents
the estimated future reward from $s$ taking action $a_i$. This network has
$\approx 4e6$ weights.}
\label{fig:dqn}
\end{figure}

\begin{figure}[!htp]
\centering
\includegraphics[width=0.8\textwidth]{img/dueling-dqn}
\caption{The Dueling Deep-Q Network (DQN) presented by \cite{dueling-deep-q}.
The densely connected model separates the data into two streams that estimate
state value and action advantage respectively.
This network has $\approx 8e6$ weights.}
\label{fig:dueling-dqn}
\end{figure}

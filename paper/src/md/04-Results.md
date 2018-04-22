# Results

We combine both \ac{DDQN} and Double Deep-$Q$ to produce the Double Dueling
Deep-$Q$ Network. We use the hyperparameters in Table
\ref{tab:hyperparameters}. Because of issues with the \ac{NES} emulator
crashing during training, we were unable to keep track of the number of
training frames $T$ and performance metrics such as the loss and reward of
each training episode. We can however, validate an untrained and trained
agent to confirm learning.

\begin{figure}[!ht]
\begin{minipage}{\textwidth}
%
\begin{minipage}{0.45\textwidth}
\includegraphics[width=\textwidth]{img/games/Breakout/initial}
\captionof*{figure}{Initial scores.}
\end{minipage}
%
\hfill
%
\begin{minipage}{0.45\textwidth}
\includegraphics[width=\textwidth]{img/games/Breakout/final}
\captionof*{figure}{Final scores.}
\end{minipage}
%
\end{minipage}
\caption{Scores of the Double \ac{DDQN} on $100$ validation games of
\textbf{Breakout} before and after training for $1e7$ frames.}
\label{fig:results-Breakout}
\end{figure}



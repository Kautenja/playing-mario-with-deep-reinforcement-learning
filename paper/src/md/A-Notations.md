# Notations

Table: Symbols and Annotations \label{tab:notation}

| Symbol        | Annotation                                                           |
|:--------------|:---------------------------------------------------------------------|
| $\mathcal{S}$ | The state space for a game as pixel space in the Y channel
| $\mathcal{A}$ | The discrete action space of button combinations for a game
| $\mathcal{R}$ | The unique reward scheme for a game
| $\phi(s)$     | A down-sampler that reduces dimensionality of $s \in \mathcal{S}$
| $x_{\phi}$    | The pixels $(x_l, x_r)$ to crop from the left and right $x$ direction
| $y_{\phi}$    | The pixels $(y_t, y_b)$ to crop from the top and bottom $y$ direction
| $k$           | The number of frames for an agent to hold each action for
| $l$           | The size of the history of frames that the agent experiences at each step
| $e$           | A memory of $e = (s, a, r, d, s')$
| $D$           | The replay memory of past experiences made by the agent
| $D'$          | A mini-batch of replay data drawn from a uniform distribution $D' \sim U(D)$
| $N$           | The total size of the replay memory
| $n$           | The mini-batch size for training the \ac{DQN}

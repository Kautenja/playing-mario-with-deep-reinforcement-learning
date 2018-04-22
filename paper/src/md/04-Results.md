# Results

We combine both \ac{DDQN} and Double Deep-$Q$ to produce the Double Dueling
Deep-$Q$ Network. We use the hyperparameters in Table
\ref{tab:hyperparameters}. Because of issues with the \ac{NES} emulator
crashing during training, we were unable to keep track of the number of
training frames $T$ and performance metrics such as the loss and reward of
each training episode. We can however, validate an untrained and trained
agent to confirm learning.

Table: Validation results of Double \ac{DDQN} before and after training to
play games for $1e7$ frames.

+---------+------------+----------+----------+------------+-----------------+
|         |   Breakout |   Enduro |     Pong |   Seaquest |   SpaceInvaders |
+=========+============+==========+==========+============+=================+
| Final   |   85.4343  |  1184.06 |  19.6364 | 263.818    |         52.9899 |
+---------+------------+----------+----------+------------+-----------------+
| Initial |   -3.43434 |     0    | -21      |   0.585859 |          6.9899 |
+---------+------------+----------+----------+------------+-----------------+

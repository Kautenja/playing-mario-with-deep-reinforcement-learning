
# Abstract

>  This paper presents a partially model-free adaptive optimal control
solution to the deterministic nonlinear discrete-time (DT) tracking control
problem in the presence of input constraints. The tracking error dynamics and
reference trajectory dynamics are first combined to form an augmented system.
Then, a new discounted performance function based on the augmented system is
presented for the optimal nonlinear tracking problem. In contrast to the
standard solution, which finds the feed-forward and feedback terms of the
control input separately, the minimization of the proposed discounted
performance function gives both feedback and feed-forward parts of the control
input simultaneously. This enables us to encode the input constraints into the
optimization problem using a non-quadratic performance function. The DT
tracking Bellman equation and tracking Hamilton–Jacobi–Bellman (HJB) are
derived. An actor–critic-based reinforcement learning algorithm is used to
learn the solution to the tracking HJB equation online without requiring
knowledge of the system drift dynamics. That is, two neural networks (NNs),
namely, actor NN and critic NN, are tuned online and simultaneously to
generate the optimal bounded control policy. A simulation example is given to
show the effectiveness of the proposed method.

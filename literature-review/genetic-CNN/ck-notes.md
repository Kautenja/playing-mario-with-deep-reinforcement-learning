
# Abstract

*   automatic search for network _structures_
*   number of structures possible increase exponentially with the number of
	layers in the network
*   proposed fixed length binary string representation of network space
*   MNIST, CIFAR-10
*   transfer learning to ILSVRC2012

> The deep Convolutional Neural Network (CNN) is the state-of-the-art solution
for large-scale visual recognition. Following basic principles such as
increasing the depth and constructing highway connections, researchers have
manually designed a lot of fixed network structures and verified their
effectiveness. In this paper, we discuss the possibility of learning deep
network structures automatically. Note that the number of possible network
structures increases exponentially with the number of layers in the network,
which inspires us to adopt the genetic algorithm to efficiently traverse this
large search space. We first propose an encoding method to represent each
network structure in a fixed-length binary string, and initialize the genetic
algorithm by generating a set of randomized individuals. In each generation,
we define standard genetic operations, e.g., selection, mutation and
crossover, to eliminate weak individuals and then generate more competitive
ones. The competitiveness of each individual is defined as its recognition
accuracy, which is obtained via training the network from scratch and
evaluating it on a validation set. We run the genetic process on two small
datasets, i.e., MNIST and CIFAR10, demonstrating its ability to evolve and
find high-quality structures which are little studied before. These structures
are also transferable to the large-scale ILSVRC2012 dataset.


# Conclusions

*   large fraction of unexplored space
	*   did not explore networks with non-convolutional modules like Maxout
	*   multi-scale model like in inception
*   _they only evolve the structure, the network is trained separately_
	*   CoDeepNEAT later goes on to do this and achieve record accuracy on MNIST

> Despite the interesting results we have obtained, our algorithm suffers from
several drawbacks. First, a large fraction of network structures are still
unexplored, including those with non-convolutional modules like Maxout [8],
and the multi-scale strategy used in the inception module [32]. Second, in the
current work, the genetic algorithm is only used to explore the network
structure, whereas the network training process is performed separately. It
would be very interesting to incorporate the genetic algorithm to training the
network structure and weights simultaneously. These directions are left for
future work.


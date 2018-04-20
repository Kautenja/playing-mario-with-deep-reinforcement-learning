"""An abstraction of a chromosome for evolving Deep Q networks."""
from copy import deepcopy
import numpy as np


class DeepQChromosome(object):
    """A chromosome representing a Deep Q network candidate."""

    # indexes of the layers that this chromosome genomes
    layer_indexes = [2, 4, 6, 9, 11]
    # the shapes of the weights
    W_shape = [
        (8, 8, 4, 32),
        (4, 4, 32, 64),
        (3, 3, 64, 64),
        (7744, 512),
        (512, 6),
    ]
    # the shapes of the biases
    b_shape = [
        (32,),
        (64,),
        (64,),
        (512,),
        (6,),
    ]

    def __init__(self,
        env, model,
        theta: list=None,
        sigma: float=0.0005,
        frames_to_play: int=20000,
    ) -> None:
        """
        Initialize a new chromosome.

        Args:
            env: the environment to validate on
            model: the model to fit weights to
            theta: the weights to initialize as the genotype
            sigma: the random value std if theta is 'random'
            frames_to_play: the number of frames to play at max

        Returns:
            None

        """
        self.env = env
        self.model = model
        # if no theta, setup a blank list
        if theta is None:
            theta = [None] * len(self.layer_indexes)
        # if random theta, randomize
        elif theta == 'random':
            theta = [None] * len(self.layer_indexes)
            for i, (W, b) in enumerate(zip(self.W_shape, self.b_shape)):
                W = sigma * np.random.uniform(0, 1, W)
                b = sigma * np.random.uniform(0, 1, b)
                theta[i] = W, b

        self.theta = theta
        self.sigma = sigma
        self.frames_to_play = frames_to_play
        self._fitness = None
        # build a mask for masking predictions from the network
        mask_shape = env.observation_space.shape[-1], env.action_space.n
        self.mask = np.ones(mask_shape)

    def __lt__(self, other) -> bool:
        """
        Return a boolean determining if this instance is < another.

        Args:
            other: the other chromosome to compare against

        Returns:
            true if self has a fitness < other

        """
        return self.fitness < other.fitness

    def __le__(self, other) -> bool:
        """
        Return a boolean determining if this instance is <= another.

        Args:
            other: the other chromosome to compare against

        Returns:
            true if self has a fitness <= other

        """
        return self.fitness <= other.fitness

    def __eq__(self, other) -> bool:
        """
        Return a boolean determining if this instance is == another.

        Args:
            other: the other chromosome to compare against

        Returns:
            true if self has a fitness == other

        """
        return self.fitness == other.fitness

    def copy(self):
        """Return a copy of this chromosome"""
        return DeepQChromosome(
            self.env,
            self.model,
            deepcopy(self.theta),
            self.sigma,
            self.frames_to_play
        )

    def get_from_model(self, model) -> None:
        """
        Fill this chromosome's genotype with weights from a model.

        Args:
            model: the model to get the weights from

        Returns:
            None

        """
        # iterate over the layers in the genotype
        for i, layer_i in enumerate(self.layer_indexes):
            # copy the weights from the model
            self.theta[i] = model.layers[layer_i].get_weights()

    def set_to_model(self, model) -> None:
        """
        Set the weights of a model to the genotype of this chromosome.

        Args:
            model: the model to set the weights

        Returns:
            None

        """
        # iterate over the layers in the genotype
        for i, layer_i in enumerate(self.layer_indexes):
            # copy the weights to the model
            model.layers[layer_i].set_weights(self.theta[i])

    def mutate(self) -> None:
        """Mutate the weights in this genotype."""
        for W, b in self.theta:
            W += self.sigma * np.random.normal(0, 1, W.shape)
            b += self.sigma * np.random.normal(0, 1, b.shape)

    @property
    def fitness(self):
        """Return the fitness of this individual."""
        if self._fitness is None:
            self._fitness = self.evaluate()
        return self._fitness

    def evaluate(self, cache_fitness: bool=False) -> float:
        """
        Evaluate the chromosome for a given env and model frame.

        Args:
            cache_fitness: whether to cache to fitness in `self.fitness`

        Returns:
            the score after 1 episode

        """
        # set the model with the weights from self
        self.set_to_model(self.model)
        # run an episode in the emulator
        done = False
        score = 0
        frames = 0
        # reset the game and get the initial state
        state = self.env.reset()
        self.env.render(mode='rgb_array')

        while not done:
            # predict an action from a stack of frames
            state = state[np.newaxis, :, :, :]
            actions = self.model.predict([state, self.mask], batch_size=1)
            action = np.argmax(actions)
            # fire the action and observe the next state, reward, and flag
            state, reward, done, _ = self.env.step(action=action)
            self.env.render(mode='rgb_array')
            score += reward
            frames += 1
            # finish early if we surpass the frame limit
            if frames >= self.frames_to_play:
                break

        if cache_fitness:
            self._fitness = score

        return score

    def evaluate_multiple(self, games: int=None) -> float:
        """
        Evaluate the chromosome for a given env and model frame multiple times.

        Args:
            repetitions: the number of times to repeat to

        Returns:
            the mean score after `repetitions` episode

        """
        # if we're repeating, go ahead and do so
        score = np.mean([self.evaluate(False) for _ in range(games)])
        # set the fitness if it is None or the current score exceeds it
        if self._fitness is None or score > self._fitness:
            self._fitness = score

        return self._fitness


__all__ = [
    DeepQChromosome.__name__
]

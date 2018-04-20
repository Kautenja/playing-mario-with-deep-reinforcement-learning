"""An implementation of Genetic Deep Reinforcement Learning."""
import numpy as np
from typing import Callable
from tqdm import tqdm
from pygame.time import Clock
from src.models import build_deep_q_model
from src.evolve.chromosome import DeepQChromosome
from .agent import Agent


class GeneticAgent(Agent):
    """The Genetic learning algorithm."""

    def __init__(self,
        env,
        render_mode: str='rgb_array',
        population_size: int=30,
        truncation_size: int=15,
        elite_repetitions: int=10,
    ) -> None:
        """
        Initialize a new Genetic Agent.

        Args:
            env: the environment to run on
            render_mode: the mode for rendering frames in the OpenAI gym env
                         -   'human': render in the emulator (default)
                         -   'rgb_array': render in the backend and return a
                                          numpy array (server/Jupyter)
            population_size: the size of the population N.

        Returns:
            None

        """
        self.env = env
        self.render_mode = render_mode
        self.population_size = population_size
        self.truncation_size = truncation_size
        self.elite_repetitions = elite_repetitions
        # build the neural model for estimating Q values
        mask_shape = (env.observation_space.shape[-1], env.action_space.n)
        self.mask = np.ones(mask_shape)
        self.model = build_deep_q_model(
            image_size=env.observation_space.shape[:2],
            num_frames=env.observation_space.shape[-1],
            num_actions=env.action_space.n
        )

    def __repr__(self) -> str:
        """Return a debugging string of this agent."""
        return """{}(env={}, render_mode={})""".lstrip().format(
            self.__class__.__name__,
            self.env,
            repr(self.render_mode),
        )

    def train(self,
        generations: int=500,
        callback: Callable=None,
    ) -> None:
        """
        Train the network for a number of episodes (games).

        Args:
            frames_to_play: the number of frames to play the game for
            callback: an optional callback to get updates about the score,
                      loss, discount factor, and exploration rate every
                      episode

        Returns:
            None

        """
        # setup the initial population of chromosomes
        population = [None] * self.population_size
        for i in range(self.population_size):
            population[i] = DeepQChromosome(self.env, self.model, 'random')
        population = sorted(population, reverse=True)

        # run for the number of generations
        for _ in tqdm(range(generations), unit='generation'):
            # take the elite as the first member and reevaluate to better
            # estimate the fitness
            self.elite = population[0]
            self.elite.evaluate(repetitions=self.elite_repetitions)
            # select parents with truncation selection
            parents = population[:self.truncation_size]
            # select the children
            children = [None] * self.population_size
            for i in range(self.population_size):
                children[i] = np.random.choice(parents).copy()
                children[i].mutate()
            # select survivors in the population
            population = sorted([self.elite] + population + children, reverse=True)
            population = population[:self.population_size]
            # pass the population to the callback
            if callable(callback):
                callback(population)

        # set the elite to self after
        self.elite.set_to_model(self.model)

    def play(self, games: int=100, fps: int=None) -> np.ndarray:
        """
        Run the agent without training for a number of games.

        Args:
            games: the number of games to play
            fps: the frame-rate to limit game play to
                - if None, the frame-rate will not be limited (i.e infinite)

        Returns:
            an array of scores, one for each game

        """
        # initialize a clock to keep the frame-rate bounded
        clock = Clock()
        # a list to keep track of the scores
        scores = np.zeros(games)
        # iterate over the number of games
        for game in tqdm(range(games), unit='game'):
            done = False
            score = 0
            # reset the game and get the initial state
            state = self._initial_state()

            while not done:
                state = state[np.newaxis, :, :, :]
                actions = self.model.predict([state, self.mask], batch_size=1)
                action = np.argmax(actions)
                # hold the action for the number of frames
                state, reward, done = self._next_state(action)
                score += reward
                # bound the frame rate if there is an fps provided
                if fps is not None:
                    clock.tick(fps)
            # push the score onto the history
            scores[game] = score

        return scores


__all__ = [
    GeneticAgent.__name__
]

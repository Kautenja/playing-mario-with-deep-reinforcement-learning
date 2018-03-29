"""A package with implementations of deep reinforcement agents."""
from .a3c_agent import A3C_Agent
from .deep_q_agent import DeepQAgent
from .double_deep_q_agent import DoubleDeepQAgent
from .random_agent import RandomAgent


# explicitly define the outward facing API of this package.
__all__ = [
    'A3C_Agent',
    'DeepQAgent',
    'DoubleDeepQAgent',
    'RandomAgent'
]

"""Hyperparameters used by previous papers in this are."""
from keras.optimizers import RMSprop
from base.annealing_variable import AnnealingVariable
from models.losses import huber_loss


# configuration for the the deep q architecture from DeepMind
deep_q = {
    'replay_memory_size': 1000000,
    'agent_history_length': 4,
    'discount_factor': 0.99,
    'update_frequency': 4,
    'optimizer': RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
    'exploration_rate': AnnealingVariable(1.0, 0.1, 1000000),
    'null_op_max': 30,
    'null_op': 0,
    'loss': huber_loss,
    'image_size': (84, 84),
    'render_mode': 'human',
}

# configuration for the the double deep q architecture (mirrors OG Deep Q)
# also from DeepMind
double_deep_q = {
    'replay_memory_size': 1000000,
    'agent_history_length': 4,
    'discount_factor': 0.99,
    'update_frequency': 4,
    'optimizer': RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
    'exploration_rate': AnnealingVariable(1.0, 0.1, 1000000),
    'null_op_max': 30,
    'null_op': 0,
    'loss': huber_loss,
    'image_size': (84, 84),
    'render_mode': 'human',
    'target_update_freq': 10000
}

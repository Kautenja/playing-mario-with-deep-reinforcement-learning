"""A command line interface to train Tiramisu vision models."""
import _top_level
from src.training import stochastic
from src.utils import log_
from src.cli import args_and_groups
from src.cli import build_argparser


# build the command line argument parser
PARSER, AGENT, TRAIN, PLAY = build_argparser()


# get the arguments from the argument parser
ARGS, ARG_GROUPS = args_and_groups(PARSER)


@stochastic(ARGS.seed, ARGS.gpu)
def train():
    """Train the model."""
    log_('Random Agent')


# train the model
train()

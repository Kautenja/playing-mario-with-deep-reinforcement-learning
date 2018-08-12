"""(Dueling/Double) Deep-Q learning for OpenAI Gym environments."""
import argparse
from .train import train
from .play import play, play_random


# mapping of command line arguments by their flags to the options they embody
_CMD_LINE_ARGS = {
    ('--env', '-e'): {
        'type': str,
        'default': 'SuperMarioBros-1-1-v2',
        'help': 'The name of the environment to play',

    },
    ('--mode', '-m'): {
        'type': str,
        'default': 'train',
        'help': 'The execution mode as either: train, play or random',
        'choices': ['train', 'play', 'random'],
    },
    ('--output', '-o'): {
        'type': str,
        'default': 'results',
        'help': 'The directory to (store/load) results (in/from)',
    },
    ('--monitor', '-M'): {
        'type': bool,
        'default': False,
        'help': 'whether to monitor the operation (record frames)',
    },
}


def _get_args() -> dict:
    """Parse command line arguments and return them."""
    # create an argument parsers with the module's docstring as the description
    parser = argparse.ArgumentParser(description=__doc__)
    # iterate over the flags and options for the command line interface
    for flags, options in _CMD_LINE_ARGS.items():
        parser.add_argument(*flags, **options)
    # parse arguments and return them
    return parser.parse_args()


def main() -> None:
    """The main entry point for the command line interface."""
    # parse arguments from the command line (argparse validates arguments)
    args = _get_args()
    # select the method for playing the game
    mode = args.mode
    if mode == 'train':
        train(
            env_id=args.env,
            output_dir=args.output,
            monitor=args.monitor,
        )
    elif mode == 'random':
        play_random(
            env_id=args.env,
            output_dir=args.output,
            monitor=args.monitor,
        )
    elif mode == 'play':
        play(
            results_dir=args.output,
            monitor=args.monitor,
        )


# explicitly define the outward facing API of this module
__all__ = [main.__name__]

"""(Dueling/Double) Deep-Q learning for OpenAI Gym environments."""
import argparse
from .train import train
from .play import play, play_random


def _get_args() -> dict:
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(description=__doc__)
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBros-1-1-v2',
        help='The name of the environment to play.'
    )
    # add the argument for the mode of execution as either human or random
    parser.add_argument('--mode', '-m',
        type=str,
        default='train',
        choices=['train', 'play', 'random'],
        help='The execution mode as either `train`, `play` or `random`.'
    )
    # add the argument for the output directory to store results in
    parser.add_argument('--output', '-o',
        type=str,
        default='results',
        help='The directory to (store/load) results (in/from).'
    )
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
            output_dir=args.output
        )
    elif mode == 'random':
        play_random(
            env_id=args.env,
            output_dir=args.output
        )
    elif mode == 'play':
        play(
            results_dir=args.output
        )


# explicitly define the outward facing API of this module
__all__ = [main.__name__]

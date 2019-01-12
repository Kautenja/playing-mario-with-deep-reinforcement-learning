"""A method for building instances of ArgumentParser."""
import argparse


def build_argparser():
    # create an argument parser to read arguments from the command line
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # create groups for different functions
    agent = parser.add_argument_group('agent')
    # create a group for the train arguments
    train = parser.add_argument_group('train')
    # create a group for the play arguments
    play = parser.add_argument_group('play')
    # MARK: Metadata
    # add an argument for the output directory to write to and read from
    parser.add_argument('--output_dir', '-o',
        type=str,
        help='The directory to (store/load) results (in/from).',
        required=False,
        default='results',
    )
    # add an argument for the dataset to use
    parser.add_argument('--gpu', '-G',
        type=int,
        help='The gpu to use if any.',
        required=False,
        default=-1,
    )
    # add an argument for the RNG seed to use
    parser.add_argument('--seed',
        type=int,
        help='The random number seed to use for NumPy, TensorFlow, and random.',
        required=False,
        default=1,
    )
    # add an argument for whether to monitor
    parser.add_argument('--monitor', '-M',
        action='store_true',
        help='whether to monitor the operation (periodically record episodes).',
        required=False,
        default=False,
    )
    # MARK: Agent
    # add an argument for the environment to play
    agent.add_argument('--env', '-e',
        type=str,
        help='The gym ID of the environment to play.',
        required=False,
        default='SuperMarioBros-1-4-v0',
    )
    # add an argument for the render mode
    agent.add_argument('--render_mode', '-R',
        type=str,
        help='Whether to use the GUI to render frames (will slow the task).',
        required=False,
        default=None,
        choices=['human', 'rgb_array'],
    )
    # MARK: play
    # add an argument for the number of games to play
    play.add_argument('--games', '-g',
        type=int,
        help='The number of games to play.',
        required=False,
        default=100,
    )

    return parser, agent, train, play


# explicitly define the outward facing API of the module
__all__ = [build_argparser.__name__]

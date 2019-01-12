"""A method for loading arguments and groups from ArgumentParser instances."""
import argparse


def args_and_groups(parser: argparse.ArgumentParser):
    """
    Return a dictionary of argument groups from an argument parser.

    Args:
        parser: the argument parser to get argument groups from

    Returns:
        a dictionary of argument groups to respective arguments

    """
    # parse the arguments from the parser
    args = parser.parse_args()
    # regroup the arguments according to their semantic groups
    arg_groups = dict()
    # iterate over the groups in the parser
    for group in parser._action_groups:
        # create a dictionary for the groups actions
        group_args = dict()
        # iterate over the actions in the group
        for action in group._group_actions:
            group_args[action.dest] = getattr(args, action.dest, None)
        # create a name-space for the arguments and store them
        arg_groups[group.title] = argparse.Namespace(**group_args)

    return args, arg_groups


# explicitly define the outward facing API of the module
__all__ = [args_and_groups.__name__]

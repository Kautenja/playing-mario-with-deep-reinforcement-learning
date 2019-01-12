"""A method for logging to the console."""
import pprint


# create a pretty print for the logger
_PRINTER = pprint.PrettyPrinter(indent=4)


def log_(header: str,
    content: str=None,
    color: str='\x1b[1;32m'
) -> None:
    """
    Log to the console.

    Args:
        header: the header to log
        content: the content to log
        color: the color to print to the console

    Returns:
        None

    """
    # print the header to the console
    print()
    print('{}{}\x1b[0m'.format(color, header))
    # if there is no content, return
    if content is None:
        return
    # print the content using the pretty printer
    _PRINTER.pprint(content)
    print()


# explicitly define the outward facing API of this module
__all__ = [log_.__name__]

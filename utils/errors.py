"""
Shorthands for printing errors and warnings, optionally exit.
"""

# Standard library imports
import sys
import typing

# Type aliases
ExitStatus = typing.Union[int, None]


def eprint(message, exit: ExitStatus = 1):
    """ Prints the given error message to stderr, optionally exit the program
        with the given error code. """
    print('error:', message, file=sys.stderr)
    if exit is not None:
        sys.exit(exit)


def warn(message, exit: ExitStatus = None):
    """ Prints the given warning message to stderr, optionally exit the program
        with the given error code. """
    print('warning:', message, file=sys.stderr)
    if exit is not None:
        sys.exit(exit)

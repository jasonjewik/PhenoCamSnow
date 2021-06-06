# Standard library imports
from pathlib import Path
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


def validate_directory(directory: str) -> Path:
    """ Validates the given directory. Prints an error message and exits
        if the argument does not exist or is not a directory. """
    try:
        direc = Path(directory)
    except:
        eprint(f'could not parse {directory} as a directory')
    if not direc.exists():
        eprint(f'{direc} does not exist')
    if not direc.is_dir():
        eprint(f'{direc} is not a directory')
    return direc


def validate_file(fname: str, extension: str, panic_on_overwrite: bool = True) -> Path:
    """ Validates the given file. Can ask for confirmation on potential
        overwrite and ask for re-input if the given file is not the correct
        extension. Extensions should include the '.' prefix. """
    while True:
        try:
            fpath = Path(fname)
        except:
            eprint(f'could not parse {fname} as a file')
        suffixes = fpath.suffixes
        if len(suffixes) == 1 and suffixes[0] == extension:
            if panic_on_overwrite and fpath.is_file() and fpath.exists():
                warn(f'{fpath} already exists')
                overwrite = input('overwrite? [y/N/q]: ')
                if overwrite.lower() == 'y':
                    break
                if overwrite.lower() == 'q':
                    warn('quit', exit=1)
                else:
                    new_csvfile = input('new csv file: ')
            else:
                break
        else:
            eprint(f'{fpath} is not a {extension} file')
    return fpath


class ProgressBar:
    """ Provides a class for displaying a progress bar. Useful for long-running tasks. """

    def __init__(self, total: int, start: int = 0, display_fraction: bool = False):
        self.idx = start
        self.total = total
        self.frac = display_fraction

    def is_complete(self) -> bool:
        """ Returns True if the progress bar's current index is equal to the
        last item. """
        return self.idx == self.total

    def inc(self, i: int = 1):
        """ An alias for increment(). """
        self.increment(i)

    def increment(self, inc: int = 1):
        """ Increments the current index by inc. Defaults to 1. """
        self.idx += inc

    def update(self, new_idx: typing.Union[int, None]):
        """ Updates the current index to new_idx. If new_idx is not given, this
        this calls increment(). """
        if new_idx is None:
            self.increment()
        else:
            self.idx = new_idx

    def display(self):
        """ Displays the progress bar. """
        complete = self.idx / self.total * 100
        complete = min(complete, 100)
        num_hashes = int(complete / 10)
        num_dots = 10 - num_hashes

        if not self.frac:
            suffix = f'{round(complete)}%'
        else:
            suffix = f'{self.idx}/{self.total}'

        hashes = '#' * num_hashes
        dots = '.' * num_dots
        output_str = f'Progress: [ {hashes}{dots} ] {suffix}'

        if self.is_complete():
            end = '\nDone!\n'
        else:
            end = '\r'

        print(output_str, sep='', end=end)

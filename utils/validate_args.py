"""
Validates command-line arguments.
"""

# Standard library imports
from pathlib import Path

# Local application imports
from utils.errors import eprint, warn


def validate_directory(directory: str) -> Path:
    """ Validates the given directory. Prints an error message and exits
        if the argument does not exist or is not a directory. """
    direc = Path(directory)
    if not direc.exists():
        eprint(f'{direc} does not exist')
    if not direc.is_dir():
        eprint(f'{direc} is not a directory')
    return direc


def validate_csv(csvfile: str, panic_on_overwrite: bool = True) -> Path:
    """ Validates the given CSV. Can ask for confirmation on potential
        overwrite and ask for re-input if the given file is not a CSV. """
    while True:
        csv = Path(csvfile)
        suffixes = csv.suffixes
        if len(suffixes) == 1 and suffixes[0] == '.csv':
            if panic_on_overwrite and csv.is_file() and csv.exists():
                warn(f'{csv} already exists')
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
            eprint(f'{csv} is not a CSV file')
    return csv

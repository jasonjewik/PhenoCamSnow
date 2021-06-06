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

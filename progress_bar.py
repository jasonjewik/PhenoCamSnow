import time
import typing


class ProgressBar:
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


if __name__ == '__main__':
    dec_pgbar = ProgressBar(100)
    while not dec_pgbar.is_complete():
        time.sleep(0.01)
        dec_pgbar.inc()
        dec_pgbar.display()

    frac_pgbar = ProgressBar(100, True)
    while not frac_pgbar.is_complete():
        time.sleep(0.01)
        frac_pgbar.inc()
        frac_pgbar.display()

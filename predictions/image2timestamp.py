# Standard library imports
import sys

# Third party imports
import pandas as pd

def extract_timestamp(x):
    stem = x.split('.')[0]
    arr = stem.split('_')[1:]
    year, month, day = arr[:3]
    hms = arr[3]
    hour, minute, second = hms[:2], hms[2:4], hms[4:]
    timestamp = f'{year}-{month}-{day} {hour}:{minute}:{second}'
    return timestamp

csv_file = sys.argv[1]
df = (pd.read_csv(csv_file, sep=',', converters={'image': extract_timestamp})
        .rename(columns={'image': 'timestamp'}))
df.to_csv(csv_file, index=False)
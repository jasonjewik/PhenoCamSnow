"""
A script for labeling image data from the canadaobj site.
"""

# Standard library imports
from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

# Third party imports
import cv2
import matplotlib.pyplot as plt

# Local application imports
from utils.errors import eprint, warn
from utils.progress_bar import ProgressBar
from utils.validate_args import validate_directory, validate_file


def main():
    # Parse and validate arguments
    parser = ArgumentParser(description='Manual image labeling tool. \
        Assumes images to be .jpg, as returned by the Phenocam download tool.')
    parser.add_argument('dir', action='store', default=Path.cwd(),
                        help='the directory containing the images to be labeled')
    parser.add_argument('-o', '--output', action='store', default='./labels.csv',
                        help='the csv file to put the labels, default is labels.csv')
    parser.add_argument('-t', '--type', action='store', default='snow',
                        help='whether to label the images by saturation or by \
                            snow, defaults to snow')
    args = parser.parse_args()
    img_dir = validate_directory(args.dir)
    out_csv = validate_file(args.output, extension='.csv')
    valid_types = ['snow', 'saturation']
    if args.type not in valid_types:
        eprint(f'{args.type} is not a valid label type', exit=None)
        eprint(f'pick from {valid_types}')

    # Label images
    print('== Instructions ==')
    if args.type is 'snow':
        print('1. press "1" to label the image as "no snow"')
        print('2. press "2" to label the image as "snow on ground"')
        print('3. press "3" to label the image as "snow on canopy"')
        print('4. press any other key to skip')
    else:
        print('1. press "1" to label the image as "high saturation"')
        print('2. press "2" to label the image as "low saturation"')
        print('3. press any other key to skip')
    print('press "q" while the image window is active to quit')

    results = []
    num_jpgs = len(sorted(img_dir.glob('*.jpg')))
    pgbar = ProgressBar(num_jpgs, display_fraction=True)
    cv2.namedWindow('window')
    valid_keys = '1,2,3,4,5,6,7,8,9,0'.split(sep=',')
    valid_keys = list(map(ord, valid_keys))

    for fp in img_dir.glob('*.jpg'):
        pgbar.display()
        fname = fp.name
        cv2.setWindowTitle('window', fname)
        img = plt.imread(str(fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img = cv2.resize(img, (w // 4, h // 4))
        cv2.imshow('window', img)
        key = cv2.waitKey(0)
        if key in valid_keys:
            ikey = int(chr(key)) - 1  # re-index at 0
            if ikey < 0:  # wraps around to max value if underflow
                ikey = 9
            results.append([fname, ikey])
        elif key == ord('q'):
            print()
            break
        pgbar.inc()
        pgbar.display()

    cv2.destroyWindow('window')

    with open(args.output, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image', 'label'])
        for r in results:
            writer.writerow(r)
    print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()

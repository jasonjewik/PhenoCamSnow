"""
A script for labeling image data from the canadaobj site.
"""

# Standard library imports
import argparse
from pathlib import Path
import sys

# Third party imports
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Local application imports
import utils


def main():
    # Parse and validate arguments
    parser = argparse.ArgumentParser(description='Manual image labeling tool. \
        Assumes images to be .jpg, as returned by the Phenocam download tool.')
    parser.add_argument('dir', action='store', default=Path.cwd(),
                        help='the directory containing the images')
    parser.add_argument('-l', '--labels', action='store',
                        help='the csv file containing the labels')
    parser.add_argument('-o', '--output', action='store',
                        help='the csv file to put the labels')
    args = parser.parse_args()
    if args.labels is not None and args.output is not None:
        utils.eprint('please specify labels if checking or output if labeling')

    img_dir = utils.validate_directory(args.dir)
    if args.labels is not None:
        labels_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)
        labels_df = pd.read_csv(labels_csv)
    if args.output is not None:
        out_csv = utils.validate_file(args.output, extension='.csv')

    # Label/verify images
    num_jpgs = len(sorted(img_dir.glob('*.jpg')))
    pgbar = utils.ProgressBar(num_jpgs, display_fraction=True)
    print('### Instructions ###')
    if args.labels is not None:
        print('1. press any key to advance')
    if args.output is not None:
        print('1. press "1" for "no snow"')
        print('2. press "2" for "snow on ground"')
        print('3. press "3" for "snow on canopy"')
        print('4. press "0" for "bad image"')
        print('5. press "q" to quit')
        print('6. press any other key to skip')

        results = []
        cv2.namedWindow('window')
        valid_keys = ['0', '1', '2', '3']
        valid_keys = list(map(ord, valid_keys))

    for fp in img_dir.glob('*.jpg'):
        pgbar.display()
        fname = fp.name
        cv2.setWindowTitle('window', fname)
        img = plt.imread(str(fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img = cv2.resize(img, (w // 4, h // 4))
        if args.labels is not None:
            label = labels_df.query(f'image == "{fname}"')['label'].to_numpy()
            if label[0] == 0:
                caption = 'bad image'
            elif label[0] == 1:
                caption = 'no snow'
            elif label[0] == 2:
                caption = 'snow'
            # elif label[0] == 3:
            #     caption = 'snow on canopy'
            img = cv2.putText(img, caption, (w // 8, h // 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                              cv2.LINE_AA)
        cv2.imshow('window', img)
        key = cv2.waitKey(0)
        if args.output is not None and key in valid_keys:
            results.append([fname, int(chr(key))])
        elif key == ord('q'):
            print()
            break
        pgbar.inc()
        pgbar.display()

    cv2.destroyWindow('window')

    if args.output is not None:
        results_df = pd.DataFrame(results).rename(
            columns={0: 'image', 1: 'label'})
        results_df.to_csv(args.output, index=False)
        print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()

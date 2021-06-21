"""
A script for labeling image data from the canadaobj site.
"""

# Standard library imports
from argparse import ArgumentParser
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
    parser = ArgumentParser(description='Manual image labeling tool. \
        Assumes images to be .jpg, as returned by the Phenocam download tool.')
    parser.add_argument('dir', action='store', default=Path.cwd(),
                        help='the directory containing the images')
    parser.add_argument('-l', '--labels', action='store',
                        help='the csv file containing the labels')
    parser.add_argument('-o', '--output', action='store',
                        help='the csv file to put the labels')
    parser.add_argument('-t', '--type', action='store', default='snow',
                        help='whether to label the images by saturation or by \
                            snow, defaults to snow')
    args = parser.parse_args()
    if args.labels is None and args.output is None:
        utils.eprint('please specify labels, output, or both')

    img_dir = utils.validate_directory(args.dir)
    out_csv = utils.validate_file(args.output, extension='.csv')
    if args.labels is not None:
        labels_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)
        labels_df = pd.read_csv(labels_csv)
    valid_types = ['snow', 'saturation']
    if args.type not in valid_types:
        utils.eprint(f'{args.type} is not a valid label type', exit=None)
        utils.eprint(f'pick from {valid_types}')

    # Label images
    print('== Instructions ==')
    if args.type is 'snow':
        print('1. press "1" to label the image as "no snow"')
        print('2. press "2" to label the image as "snow on ground"')
        print('3. press "3" to label the image as "snow on canopy"')
        print('4. press the space bar to skip')
    else:
        print('1. press "1" to label the image as "high saturation"')
        print('2. press "2" to label the image as "low saturation"')
        print('3. press the space bar to skip')
    print('press "q" while the image window is active to quit')

    results = []
    num_jpgs = len(sorted(img_dir.glob('*.jpg')))
    pgbar = utils.ProgressBar(num_jpgs, display_fraction=True)
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
        if args.labels is not None:
            label = labels_df[labels_df['image'] == fname]['label'].to_numpy()
            if len(label) == 0:
                caption = 'bad image'
            elif label[0] == 0:
                caption = 'no snow'
            elif label[0] == 1:
                caption = 'snow'
            img = cv2.putText(img, caption, (w // 8, h // 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                              cv2.LINE_AA)
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

    results_df = pd.DataFrame(results).rename(columns={0: 'image', 1: 'label'})
    if args.labels is not None:
        labels_df = labels_df.rename(columns={'label': 'predicted'})
        results_df = pd.merge(results_df, labels_df)
    results_df.to_csv(args.output, index=False)
    print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()

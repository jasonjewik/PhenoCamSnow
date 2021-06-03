"""
A script to load and run a Nu SVC on image data. Make sure that
the number of clusters selected for K-Means matches the number
of clusters the SVC expects.
"""

# Standard library imports
from argparse import ArgumentParser
import csv
import joblib
import os
from pathlib import Path
import sys

# Third party imports
import numpy as np
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Local application imports
from utils.progress_bar import ProgressBar
from quantize_images import sample_image, scale_image


def main():
    parser = ArgumentParser(description='Classifies the given images.')
    parser.add_argument('model', action='store',
                        help='the model to use')
    parser.add_argument('dir', action='store', help='the directory \
        containing the images to be classified')
    parser.add_argument('-o', '--output', action='store',
                        default='results.csv',
                        help='where to store the results')
    parser.add_argument('-k', action='store', default=4, type=int,
                        help='the number of clusters for KMeans, default is 4')
    parser.add_argument('--stable', action='store_true', default=False,
                        help='if enabled, any random states will be fixed')

    args = parser.parse_args()
    inpath = Path(args.dir).resolve()
    modelpath = Path(args.model).resolve()
    outpath = Path(args.output).resolve()

    if not modelpath.exists():
        print(f'error: {modelpath} does not exist', file=sys.stderr)
        sys.exit(1)
    if not modelpath.is_file() or modelpath.suffix != '.joblib':
        print(f'error: {modelpath} is not a valid model file')
        sys.exit(1)

    if not inpath.exists():
        print(f'error: {inpath} does not exist', file=sys.stderr)
        sys.exit(1)
    if not inpath.is_dir():
        print(f'error: {inpath} is not a directory', file=sys.stderr)
        sys.exit(1)

    if outpath.exists() and not outpath.is_file():
        print(f'error: {outpath} is not a file', file=sys.stderr)
        sys.exit(1)

    fpaths = inpath.glob('*.jpg')
    if args.stable:
        kmeans = KMeans(n_clusters=args.k, random_state=0)
    else:
        kmeans = KMeans(n_clusters=args.k)

    all_files = os.listdir(inpath)
    num_jpgs = 0
    for fi in all_files:
        if fi.endswith('.jpg'):
            num_jpgs += 1

    clf = joblib.load(modelpath)
    pgbar = ProgressBar(num_jpgs)
    fnames = []
    predictions = []

    for fp in fpaths:
        pgbar.display()
        im = scale_image(plt.imread(fp), 0.25) / 255
        w, h, _ = im.shape
        clusters = kmeans.fit(sample_image(im, stable=args.stable))
        rgb_centers = clusters.cluster_centers_
        hsv_centers = rgb_to_hsv(rgb_centers)
        features = hsv_centers[hsv_centers[:, 2].argsort()]
        X = features.reshape((1, -1))
        fnames.append(fp.name)
        predictions.append(clf.predict(X)[0])
        pgbar.inc()

    proceed = True
    if outpath.exists() and outpath.is_file():
        print(f'warning: {outpath} already exists.')
        key = input('overwrite? [y/N] ')
        key = key.lower()
        if key == 'y':
            proceed = True
        else:
            proceed = False

    if not proceed:
        print('warning: will not check again for overwrite')
        out_name = input('provide a new name for the output file: ')
        outpath = Path(out_name).resolve()

    with open(outpath, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for fname, prediction in zip(fnames, predictions):
            timestamp = '-'.join(fname.split('_')[1:]).split('.')[0]
            writer.writerow([fname, timestamp, prediction])
    print(f'Wrote results to {outpath}')


if __name__ == '__main__':
    main()

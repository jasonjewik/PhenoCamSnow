"""
A script for performing color quantization via K-Means. Images to be quantized
must be in the same directory.
"""

# Standard library imports
import argparse
import csv
import os
from pathlib import Path
import sys

# Third party imports
import cv2
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Local application imports
import utils


def main():
    parser = argparse.ArgumentParser(
        description='Performs color quantization on the given images.')

    parser.add_argument('dir', action='store', nargs='+',
                        help='the directories containing the images to be labeled')
    parser.add_argument('-o', '--output', action='store',
                        default='features.csv',
                        help='name for the output file containing per \
                            image features')
    args = parser.parse_args()

    imdirs = [utils.validate_directory(direc) for direc in args.dir]
    outpath = utils.validate_file(args.output, extension='.csv')

    num_images = 0
    for direc in imdirs:
        num_images += len(sorted(direc.glob('*.jpg')))
    pgbar = utils.ProgressBar(num_images)
    kmeans = KMeans(n_clusters=4)
    quantized_features = {}

    try:
        for direc in imdirs:
            for fp in direc.glob('*.jpg'):
                pgbar.display()

                im = plt.imread(fp)
                im = preprocess(im, scale=0.25)
                ftrs = quantize(kmeans, im)
                quantized_features[fp.name] = ftrs.flatten()

                pgbar.inc()
                pgbar.display()
    except:
        utils.warn('Encountered error (or maybe interrupt?)')
        utils.warn('Writing current progress out to file')

    column_names = ['h0', 's0', 'v0',  # 'r0',
                    'h1', 's1', 'v1',  # 'r1',
                    'h2', 's2', 'v2',  # 'r2',
                    'h3', 's3', 'v3']  # 'r3']
    df = (pd.DataFrame(quantized_features)
            .transpose()
            .rename(columns=dict(enumerate(column_names))))
    df.to_csv(outpath, index_label='image')
    print(f'Wrote results to {outpath}')


def preprocess(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    img = img[:-30, :, :]
    h, w, _ = img.shape
    new_shape = (round(w * scale), round(h * scale))
    resized = cv2.resize(img, new_shape)
    result = resized / 255
    return result


def quantize(kmeans: KMeans, im: np.ndarray):
    h, w, c = im.shape
    unrolled_im = im.reshape((h * w, c))
    sample = shuffle(unrolled_im)[:1000]
    clusters = kmeans.fit(sample)

    rgb_centers = clusters.cluster_centers_
    hsv_centers = rgb_to_hsv(rgb_centers)
    hsv_centers = hsv_centers[hsv_centers[:, 2].argsort()]

    """ Unused code. Removing this massively speeds up the quantization.
        Plus, the classifier doesn't use ratio information anyway. """
    # labels = kmeans.predict(unrolled_im)
    # counts = [0] * kmeans.n_clusters
    # for i in range(kmeans.n_clusters):
    #     counts[i] = np.sum(np.where(labels == i, 1, 0))
    # ratios = np.array(counts) / len(labels)
    # ratios = ratios.reshape((1, ratios.shape[0])).T

    # features = np.hstack((hsv_centers, ratios))
    features = hsv_centers
    return features


if __name__ == '__main__':
    main()

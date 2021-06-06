"""
A script for performing color quantization via K-Means. Images to be quantized
must be in the same directory.
"""

# Standard library imports
from argparse import ArgumentParser
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
    parser = ArgumentParser(
        description='Performs color quantization on the given images.')

    parser.add_argument('dir', action='store',
                        help='the directory containing the images to be labeled')
    parser.add_argument('-o', '--output', action='store',
                        default='features.csv',
                        help='name for the output file containing per \
                            image features')
    parser.add_argument('-k', action='store', default=4, type=int,
                        help='the number of clusters for KMeans, default is 4')
    parser.add_argument('--save', action='store', metavar='DIR',
                        help='if enabled, will save the images to the given \
                            directory')
    args = parser.parse_args()

    imdir = utils.validate_directory(args.dir)
    outpath = utils.validate_file(args.output, extension='.csv')
    if args.save is not None:
        save_dir = utils.validate_directory(args.save)

    pgbar = utils.ProgressBar(len(sorted(imdir.glob('*.jpg'))))
    kmeans = KMeans(n_clusters=args.k)
    names = []
    cluster_centers = []
    color_ratios = []

    try:
        for fp in imdir.glob('*.jpg'):
            pgbar.display()

            im = plt.imread(fp)
            im = preprocess(im, scale=0.25)
            w, h, _ = im.shape
            rgb_centers, hsv_centers, labels, ratios = quantize(kmeans, im)
            names.append(fp.name)
            cluster_centers.append(hsv_centers)
            color_ratios.append(ratios)

            if args.save is not None:
                im = recreate(rgb_centers, labels, w, h)
                outfile = save_dir.joinpath(fp.name)
                plt.imsave(outfile, im)

            pgbar.inc()
            pgbar.display()
    except:
        utils.warn('Encountered error (or maybe interrupt?)')
        utils.warn('Writing current progress out to file')

    cluster_centers = np.asarray(cluster_centers)
    color_ratios = np.asarray(color_ratios)
    df = pd.DataFrame({
        'image': names,
        'c0_h': cluster_centers[:, 0, 0],
        'c0_s': cluster_centers[:, 0, 1],
        'c0_v': cluster_centers[:, 0, 2],
        'c0_r': color_ratios[:, 0],
        'c1_h': cluster_centers[:, 1, 0],
        'c1_s': cluster_centers[:, 1, 1],
        'c1_v': cluster_centers[:, 1, 2],
        'c1_r': color_ratios[:, 1],
        'c2_h': cluster_centers[:, 2, 0],
        'c2_s': cluster_centers[:, 2, 1],
        'c2_v': cluster_centers[:, 2, 2],
        'c2_r': color_ratios[:, 2],
        'c3_h': cluster_centers[:, 3, 0],
        'c3_s': cluster_centers[:, 3, 1],
        'c3_v': cluster_centers[:, 3, 2],
        'c3_r': color_ratios[:, 3],
    })
    df.to_csv(outpath, index=False)
    print(f'Wrote results to {outpath}')


def unroll(im: np.ndarray) -> np.ndarray:
    shape = im.shape
    return im.reshape((shape[0] * shape[1], shape[2]))


def sample(im: np.ndarray, amount: int = 1000) -> np.ndarray:
    pixel_array = unroll(im)
    result = shuffle(pixel_array)[:amount]
    return result


def recreate(codebook: np.ndarray, labels: np.ndarray, w: int, h: int) -> np.ndarray:
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def count_colors(num_colors: int, labels: np.ndarray) -> np.ndarray:
    counts = [0] * num_colors
    for i in range(num_colors):
        counts[i] = np.sum(np.where(labels == i, labels, 0))
    ratios = np.asarray(counts) / len(labels)
    return ratios


def preprocess(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    # Crops out the bottom bar of the image
    img = img[:-30, :, :]
    h, w, _ = img.shape
    new_shape = (round(w * scale), round(h * scale))
    resized = cv2.resize(img, new_shape)
    result = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255
    return result


def quantize(kmeans: KMeans, im: np.ndarray):
    clusters = kmeans.fit(sample(im))

    rgb_centers = clusters.cluster_centers_
    hsv_centers = rgb_to_hsv(rgb_centers)
    hsv_centers = hsv_centers[hsv_centers[:, 2].argsort()]

    labels = kmeans.predict(unroll(im))
    ratios = count_colors(kmeans.n_clusters, labels)

    return [rgb_centers, hsv_centers, labels, ratios]


if __name__ == '__main__':
    main()

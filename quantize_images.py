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
import numpy as np
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Local application imports
from utils.progress_bar import ProgressBar


def main():
    parser = ArgumentParser(
        description='Performs color quantization on the given images.')

    parser.add_argument('-d', '--dir', action='store', default=Path.cwd(),
                        help='the directory containing the images to be labeled, \
                            defaults to the current working directory')
    parser.add_argument('-o', '--output', action='store', default=None,
                        help='the directory to put the quantized images, and the \
                            CSV containing per image cluster centers (HSV)')
    parser.add_argument('-n', '--name', action='store',
                        default='cluster_centers.csv',
                        help='file name for the output file')
    parser.add_argument('--save', action='store_true', default=False,
                        help='if enabled, will save the images to the output \
                            directory')
    parser.add_argument('-k', action='store', default=4, type=int,
                        help='the number of clusters for KMeans, default is 4')
    parser.add_argument('--stable', action='store_true', default=False,
                        help='if enabled, any random states will be fixed')
    args = parser.parse_args()
    inpath = Path(args.dir).resolve()
    outpath = Path(args.output).resolve()

    if not inpath.exists():
        print(f'error: {inpath} does not exist', file=sys.stderr)
        sys.exit(1)
    if not inpath.is_dir():
        print(f'error: {inpath} is not a directory', file=sys.stderr)
        sys.exit(1)

    if not outpath.exists():
        outpath.mkdir()

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

    pgbar = ProgressBar(num_jpgs)
    image_features = []

    for fp in fpaths:
        pgbar.display()

        # Fit K-Means classifer
        im = scale_image(plt.imread(fp), 0.25) / 255
        w, h, _ = im.shape
        clusters = kmeans.fit(sample_image(im, stable=args.stable))

        # Each image is paired with its cluster centers in HSV format,
        # sorted by V
        rgb_centers = clusters.cluster_centers_
        hsv_centers = rgb_to_hsv(rgb_centers)
        features = hsv_centers[hsv_centers[:, 2].argsort()]
        image_features.append([fp.name, features])

        # Write the image to file, if specified
        if args.save:
            labels = kmeans.predict(unroll_image(im))
            im = recreate_image(rgb_centers, labels, w, h)
            outfile = outpath.joinpath(fp.name)
            plt.imsave(outfile, im)

        pgbar.inc()

    outfile = outpath.joinpath(args.name)
    with open(outfile, 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for im_name, ftrs in image_features:
            writer.writerow([im_name, *ftrs])
    print(f'Wrote per image cluster centers to {outfile}')


def unroll_image(im: np.ndarray) -> np.ndarray:
    shape = im.shape
    return im.reshape((shape[0] * shape[1], shape[2]))


def sample_image(im: np.ndarray, stable: bool) -> np.ndarray:
    pixel_array = unroll_image(im)
    if stable:
        result = shuffle(pixel_array, random_state=0)[:1000]
    else:
        result = shuffle(pixel_array)[:1000]
    return result


def recreate_image(codebook: np.ndarray, labels: np.ndarray, w: int, h: int) -> np.ndarray:
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def scale_image(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w, _ = img.shape
    new_shape = (round(w * scale), round(h * scale))
    resized = cv2.resize(img, new_shape)
    result = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return result


if __name__ == '__main__':
    main()

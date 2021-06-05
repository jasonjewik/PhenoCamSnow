"""
A script for computing the mean and variance of an image's saturation.
"""

# Standard library imports
from argparse import ArgumentParser
from pathlib import Path
import sys

# Third party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
from utils.errors import eprint, warn
from utils.progress_bar import ProgressBar
from utils.validate_args import validate_csv, validate_directory


def sat_meanvar(rgb_im: np.ndarray) -> [np.float64, np.float64]:
    """ Converts the given image to grayscale and HSV, returning the mean and
        variance of the image created by taking the absolute difference 
        between the grayscale image and the V part of the HSV image. """
    hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2HSV)
    s_im = hsv_im[:, :, 1]
    mean = np.mean(s_im)
    var = np.var(s_im)
    return [mean, var]


def main():
    # Parse and validate arguments
    parser = ArgumentParser(description='Computes the absolute gravy mean and \
        variance for a folder of images.')
    parser.add_argument('images', action='store',
                        help='path to the image folder, images must be jpg')
    parser.add_argument('-o', '--output', action='store', default='output.csv',
                        help='what to call the output file (must be CSV)')
    args = parser.parse_args()

    image_dir = validate_directory(args.images)
    csv_file = validate_csv(args.output)

    # Compute mean, var and write to file
    names = []
    means = []
    variances = []
    pgbar = ProgressBar(len(sorted(image_dir.glob('*.jpg'))))
    try:
        for im_path in image_dir.glob('*.jpg'):
            pgbar.display()
            im = plt.imread(im_path)
            mean, var = sat_meanvar(im)
            names.append(im_path.name)
            means.append(mean)
            variances.append(var)
            pgbar.inc()
            pgbar.display()
    except Exception as e:
        warn('Encountered error (or maybe interrupt?)')
        warn('Writing current progress out to file')

    df = pd.DataFrame({
        'image': names,
        'mean': means,
        'variance': variances
    })
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    main()

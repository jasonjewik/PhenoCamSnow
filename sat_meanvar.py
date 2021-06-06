"""
A script for computing the mean and variance of an image's saturation.
"""

# Standard library imports
from argparse import ArgumentParser
from pathlib import Path
import sys
import typing

# Third party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
import utils


def sat_meanvar(rgb_im: np.ndarray) -> [np.float64, np.float64]:
    """ Converts the given image to grayscale and HSV, returning the mean and
        variance of the image created by taking the absolute difference 
        between the grayscale image and the V part of the HSV image. """
    hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2HSV)
    s_im = hsv_im[:, :, 1]
    return [np.mean(s_im), np.var(s_im)]


def main():
    # Parse and validate arguments
    parser = ArgumentParser(description='Computes the absolute gravy mean and \
        variance for a folder of images.')
    parser.add_argument('images', action='store',
                        help='path to the image folder, images must be jpg')
    parser.add_argument('--num', action='store', type=int,
                        help='the number of pixels to sample for computating \
                            the mean and variance of image saturation, \
                                defaults to all pixels')
    parser.add_argument('-o', '--output', action='store', default='output.csv',
                        help='what to call the output file (must be csv)')
    args = parser.parse_args()

    image_dir = utils.validate_directory(args.images)
    csv_file = utils.validate_file(args.output, extension='.csv')

    # Compute mean, var and write to file
    names = []
    means = []
    variances = []
    pgbar = utils.ProgressBar(len(sorted(image_dir.glob('*.jpg'))))
    try:
        for im_path in image_dir.glob('*.jpg'):
            pgbar.display()
            im = plt.imread(im_path)
            names.append(im_path.name)
            mean, var = sat_meanvar(im)
            means.append(mean)
            variances.append(var)
            pgbar.inc()
            pgbar.display()
    except:
        utils.warn('Encountered error (or maybe interrupt?)')
        utils.warn('Writing current progress out to file')

    df = pd.DataFrame({
        'image': names,
        'mean': means,
        'variance': variances
    })
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    main()

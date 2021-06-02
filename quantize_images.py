from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from pathlib import Path
import numpy as np
from progress_bar import ProgressBar
import argparse
import matplotlib.pyplot as plt
import os
import cv2
import sys
import csv
from matplotlib.colors import rgb_to_hsv

parser = argparse.ArgumentParser(description='Quantizes the given images.')
parser.add_argument('-d', '--dir', action='store', default=Path.cwd(),
                    help='the directory containing the images to be labeled')
parser.add_argument('-o', '--output', action='store', default=None,
                    help='the directory to put the quantized images, and the \
                        CSV containing per image cluster centers (HSV) and \
                            color ratios')
parser.add_argument('--save', action='store_true', default=False,
                    help='if enabled, will save the images to the output \
                        directory')
parser.add_argument('-k', action='store', default=2, type=int,
                    help='the number of clusters for KMeans')
parser.add_argument('--stable', action='store_true', default=False,
                    help='if enabled, will fix the random states')
args = parser.parse_args()
inpath = Path(args.dir).resolve()
outpath = Path(args.output).resolve()

if not inpath.exists():
    print(f'error: {inpath} does not exist', file=sys.stderr)
    sys.exit(1)

if not outpath.exists():
    outpath.mkdir()


def unroll_pixels(im):
    shape = im.shape
    return im.reshape((shape[0] * shape[1], shape[2]))


def sample_image(im):
    pixel_array = unroll_pixels(im)
    if args.stable:
        result = shuffle(pixel_array, random_state=0)[:1000]
    else:
        result = shuffle(pixel_array)[:1000]
    return result


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def scale_image(img, scale=0.5):
    h, w, _ = img.shape
    new_shape = (round(w * scale), round(h * scale))
    resized = cv2.resize(img, new_shape)
    result = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return result


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
    im = scale_image(plt.imread(fp), 0.25) / 255
    w, h, _ = im.shape
    clusters = kmeans.fit(sample_image(im))

    # extract information about how much of the image is this color
    labels = kmeans.predict(unroll_pixels(im))
    counts = [0] * args.k
    for l in labels:
        counts[l] += 1
    ratios = np.asarray(counts) / len(labels)
    ratios = np.reshape(ratios, (len(ratios), 1))

    rgb_centers = clusters.cluster_centers_
    hsv_centers = rgb_to_hsv(rgb_centers)
    features = hsv_centers
    # features = np.append(hsv_centers, ratios, axis=1)
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/2828121#2828121
    # sorts the hsv centers by value
    features = features[features[:, 2].argsort()]
    # scale the hsv centers by the amount they show up in the image
    features = hsv_centers * ratios
    image_features.append([fp.name, features])

    if args.save:
        im = recreate_image(rgb_centers, labels, w, h)
        outfile = outpath.joinpath(fp.name)
        plt.imsave(outfile, im)

    pgbar.inc()

outfile = outpath.joinpath('cluster_centers.csv')
with open(outfile, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for im_name, ftrs in image_features:
        writer.writerow([im_name, *ftrs])
print(f'Wrote per image cluster centers to {outfile}')

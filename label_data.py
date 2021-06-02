import cv2
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import os
import csv
from progress_bar import ProgressBar


parser = argparse.ArgumentParser(description='Manual image labeling tool. \
    For each image, pick from three possible labels: no snow, snow on \
        ground, snow on canopy. Assumes images to be .jpg, as returned by the \
            Phenocam download tool.')
parser.add_argument('-d', '--dir', action='store', default=Path.cwd(),
                    help='the directory containing the images to be labeled')
parser.add_argument('-o', '--output', action='store', default='./labels.csv',
                    help='the CSV file to put the labels')
args = parser.parse_args()
dirpath = Path(args.dir).resolve()

if not dirpath.exists():
    print(f'error: {dirpath} does not exist', file=sys.stderr)
    sys.exit(1)

print('== Instructions ==')
print('1. press "1" to label the image as "no snow"')
print('2. press "2" to label the image as "snow on ground"')
print('2. press "3" to label the image as "snow on canopy"')
print('3. press "Q" to quit')
print('4. press any other key to skip the image')

results = []
all_files = os.listdir(dirpath)
num_jpgs = 0
for a in all_files:
    if a.endswith('.jpg'):
        num_jpgs += 1

fpaths = dirpath.glob('*.jpg')
cv2.namedWindow('window')
pgbar = ProgressBar(num_jpgs, display_fraction=True)

for fp in fpaths:
    pgbar.display()

    fname = fp.name
    cv2.setWindowTitle('window', fname)
    # using matplotlib because opencv fails to read some of the files
    img = plt.imread(str(fp))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, c = img.shape
    img = cv2.resize(img, (w // 4, h // 4))
    cv2.imshow('window', img)

    key = cv2.waitKey(0)
    if key == ord('1'):
        results.append([fname, 0])
    elif key == ord('2'):
        results.append([fname, 1])
    elif key == ord('3'):
        results.append([fname, 2])
    elif key == ord('q'):
        print()
        break

    pgbar.inc()

cv2.destroyWindow('window')

with open(args.output, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for r in results:
        writer.writerow(r)
print(f'Wrote results to {args.output}')

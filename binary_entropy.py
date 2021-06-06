import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


direc = Path(sys.argv[1])
images = []
entropy1 = []
entropy2 = []
for fp in direc.glob('*.jpg'):
    im = plt.imread(fp)
    h, w, _ = im.shape
    im = cv2.resize(im, (w // 4, h // 4))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    images.append(fp.name)
    entropy1.append(entropy(th))
    gray = gray[80:, :]
    blur = cv2.blur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    entropy2.append(entropy(th))

df = pd.DataFrame({
    'image': images,
    'entropy1': entropy1,
    'entropy2': entropy2
})

df.to_csv('results.csv')

"""
A script to train/evaluate/run a Nu SVM that classifies the image data from the
canadaobj site. Classifies each image as "no snow" or "has snow".
"""

# Standard library imports
from argparse import ArgumentParser
import csv
import joblib
from pathlib import Path
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

# Local application imports
import utils
from quantize_images import preprocess, sample, quantize


def main():
    parser = ArgumentParser(
        description='trains/evaluates/runs a snow classifier')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train a new model: labels, features, and output \
                            must be given; k is optional')
    parser.add_argument('--eval', action='store', metavar='MODEL',
                        help='evaluates the given model: labels, images \
                            must be given; output, k optional')
    parser.add_argument('--predict', action='store', metavar='MODEL',
                        help='classifies images using the given model: \
                            images and output must be given; k is optional')
    parser.add_argument('--features', action='store',
                        help='the csv file containing the image features (see \
                            quantize_images.py)')
    parser.add_argument('-k', action='store', default=4, type=int,
                        help='the number of clusters for KMeans, default is 4')
    parser.add_argument('--labels', action='store',
                        help='the csv file containing the image labels (must \
                            match the format returned by label_data.py')
    parser.add_argument('--images', action='store', nargs='+',
                        help='the path to the image director(ies)')
    parser.add_argument('-o', '--output', action='store',
                        help='the name of the output file')
    args = parser.parse_args()

    run_training = int(args.train)
    run_eval = int(args.eval is not None)
    run_predict = int(args.predict is not None)

    if run_training + run_eval + run_predict > 1:
        utils.eprint('specify just one of train, eval, or predict')

    elif args.train:
        features_csv = utils.validate_file(
            args.features, extension='.csv', panic_on_overwrite=False)
        label_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)

        features_df = pd.read_csv(features_csv)
        label_df = pd.read_csv(label_csv)
        joint_df = features_df.merge(label_df)
        columns = [f'c{i}_h' for i in range(args.k)]
        sats = [f'c{i}_s' for i in range(args.k)]
        vals = [f'c{i}_v' for i in range(args.k)]
        columns.extend(sats)
        columns.extend(vals)
        columns.sort()
        X = joint_df[columns].to_numpy()
        y = joint_df['label'].to_numpy()
        y = binarize(y)

        train(X, y, args.output)

    elif args.eval is not None:
        model_path = utils.validate_file(
            args.eval, extension='.joblib', panic_on_overwrite=False)
        label_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)
        image_dirs = [utils.validate_directory(direc) for direc in args.images]

        model = joblib.load(model_path)
        label_df = pd.read_csv(label_csv)

        evaluate(model, image_dirs, label_df, args.k, args.output)

    elif args.predict is not None:
        model_path = utils.validate_file(
            args.predict, extension='.joblib', panic_on_overwrite=False)
        image_dirs = [utils.validate_directory(direc) for direc in args.images]
        outpath = utils.validate_file(args.output, extension='.csv')

        model = joblib.load(model_path)
        df = predict(model, image_dirs[0], args.k)
        df.to_csv(outpath, index=False)

    else:
        utils.eprint('you need to specify one of train, eval, or predict')


def train(X: np.ndarray, y: np.ndarray, outpath: str, test_size: float = 0.2):
    """ Trains and saves a classification model. """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    clf = make_pipeline(
        StandardScaler(),
        NuSVC(nu=0.1, class_weight='balanced')
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f'Accuracy Score: {acc}')
    print(f'F1 Score: {f1}')

    outpath = utils.validate_file(outpath, extension='.joblib')
    with open(outpath, 'wb') as f:
        joblib.dump(clf, f)
    print(f'Saved model to {outpath}')


def evaluate(model: NuSVC, image_dirs: [Path], labels: pd.DataFrame, k: int, output: str):
    """ Evaluates the given classification model. """
    features = []
    indices_to_drop = []
    pgbar = utils.ProgressBar(len(labels))
    kmeans = KMeans(n_clusters=k)

    print('processing images')
    for idx, im_name in enumerate(labels['image']):
        pgbar.display()
        im_path = None
        for direc in image_dirs:
            temp_path = direc.joinpath(im_name)
            if temp_path.exists():
                im_path = temp_path
                break
        if im_path is None:
            utils.warn(f'{im_name} could not be found, skipping')
            indices_to_drop.append(idx)
        else:
            im = plt.imread(im_path)
            im = preprocess(im, scale=0.25)
            hsv_centers = quantize(kmeans, im)[1]
            h, w = hsv_centers.shape
            flattened_hsv_centers = hsv_centers.reshape((h * w))
            features.append(flattened_hsv_centers)
        pgbar.inc()
        pgbar.display()

    y_true = labels['label'].drop(indices_to_drop)
    y_true = binarize(y_true)
    y_pred = model.predict(features)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')

    if output is not None:
        output = utils.validate_file(output, extension='.csv')
        df = pd.DataFrame({
            'image': labels['image'].drop(indices_to_drop),
            'label': y_pred
        })
        df.to_csv(output, index=False)


def predict(model: NuSVC, image_dir: Path, k: int) -> pd.DataFrame:
    """ Output predicted class for each image in the given directory. """
    im_names = []
    features = []
    pgbar = utils.ProgressBar(len(sorted(image_dir.glob('*.jpg'))))
    kmeans = KMeans(n_clusters=k)

    print('processing images')
    for fp in image_dir.glob('*.jpg'):
        pgbar.display()
        im = plt.imread(fp)
        im_names.append(fp.name)
        hsv_centers = quantize(kmeans, im)[1]
        h, w = hsv_centers.shape
        flattened_hsv_centers = hsv_centers.reshape((h * w))
        features.append(flattened_hsv_centers)
        pgbar.inc()
        pgbar.display()

    predictions = model.predict(features)
    df = pd.DataFrame({
        'image': im_names,
        'label': predictions
    })
    return df


def binarize(labels: np.ndarray) -> np.ndarray:
    """ Simplifies labels so that the first class stays the same, and the rest
        are combined into one class. """
    binarized_labels = np.where(labels == 0, labels, 1)
    return binarized_labels


if __name__ == '__main__':
    main()

"""
A script to train/evaluate/run a Nu SVM that classifies the image data from the
canadaobj site. Classifies each image as "no snow" or "has snow".
"""

# Standard library imports
import argparse
import csv
import joblib
from pathlib import Path
import typing

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC

# Local application imports
import utils
from quantize_images import preprocess, quantize


def main():
    parser = argparse.ArgumentParser(
        description='trains/evaluates/runs a snow classifier')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train a new model: labels, features, and output \
                            must be given')
    parser.add_argument('--eval', action='store', metavar='MODEL',
                        help='evaluates the given model: labels, \
                            features must be given')
    parser.add_argument('--predict', action='store', metavar='MODEL',
                        help='classifies images using the given model: \
                            images and output must be given')
    parser.add_argument('--features', action='store',
                        help='the csv file containing the image features (see \
                            quantize_images.py)')
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
        columns = ['h0', 'h1', 'h2', 'h3',
                   's0', 's1', 's2', 's3',
                   'v0', 'v1', 'v2', 'v3']
        #    'r0', 'r1', 'r2', 'r3']
        X = joint_df[columns].to_numpy()
        y = joint_df['label'].to_numpy()
        y = np.where(y == 3, 2, y)
        train(X, y, args.output)

    elif args.eval is not None:
        model_path = utils.validate_file(
            args.eval, extension='.joblib', panic_on_overwrite=False)
        label_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)
        features_csv = utils.validate_file(
            args.features, extension='.csv', panic_on_overwrite=False)

        model = joblib.load(model_path)
        features_df = pd.read_csv(features_csv)
        label_df = pd.read_csv(label_csv)
        joint_df = features_df.merge(label_df)
        columns = ['h0', 'h1', 'h2', 'h3',
                   's0', 's1', 's2', 's3',
                   'v0', 'v1', 'v2', 'v3']
        """ The unusued ratio information: """
        #    'r0', 'r1', 'r2', 'r3']
        X = joint_df[columns].to_numpy()
        y_true = joint_df['label'].to_numpy()
        y_true = np.where(y_true == 3, 2, y_true)
        evaluate(model, X, y_true)

    elif args.predict is not None:
        model_path = utils.validate_file(
            args.predict, extension='.joblib', panic_on_overwrite=False)
        image_dirs = [utils.validate_directory(direc) for direc in args.images]
        outpath = utils.validate_file(args.output, extension='.csv')

        model = joblib.load(model_path)
        df = None
        for direc in image_dirs:
            temp_df = predict(model, image_dir=direc)
            df = temp_df if df is None else df.append(temp_df)
        df.to_csv(outpath, index=False)
        print('Predictions Preview:')
        print(df)

    else:
        utils.eprint('you need to specify one of train, eval, or predict')


def train(X: np.ndarray, y: np.ndarray, outpath: str, test_size: float = 0.33):
    """ Trains and saves a classification model. """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    clf = NuSVC(nu=0.2, class_weight='balanced')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None)
    print(f'Accuracy Score: {acc}')
    print(f'F1 Score: {f1}')

    outpath = utils.validate_file(outpath, extension='.joblib')
    with open(outpath, 'wb') as f:
        joblib.dump(clf, f)
    print(f'Saved model to {outpath}')


def evaluate(model: NuSVC, X: np.ndarray, y_true: np.ndarray):
    """ Evaluates the given classification model. """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')


def predict(model: NuSVC, image_dir: Path = None, single_image: Path = None) -> pd.DataFrame:
    """ Output predicted class for each image in the given directory, or for
        a single image. """
    kmeans = KMeans(n_clusters=4)

    if image_dir is not None:
        im_names = []
        X = []
        pgbar = utils.ProgressBar(len(sorted(image_dir.glob('*.jpg'))))

        print('processing images')
        for fp in image_dir.glob('*.jpg'):
            pgbar.display()
            im = plt.imread(fp)
            im = preprocess(im, scale=0.25)
            im_names.append(fp.name)
            ftrs = quantize(kmeans, im)
            X.append(ftrs.T.flatten())
            pgbar.inc()
            pgbar.display()
        X = np.array(X)
        predictions = model.predict(X)
        df = pd.DataFrame({
            'image': im_names,
            'label': predictions
        })
    elif single_image is not None:
        im = plt.imread(single_image)
        X = quantize(kmeans, im).T.flatten().reshape((1, -1))
        prediction = model.predict(X)
        df = pd.DataFrame({
            'image': single_image.name,
            'label': prediction
        })
    else:
        utils.eprint('specify a directory or an image')
    return df


if __name__ == '__main__':
    main()

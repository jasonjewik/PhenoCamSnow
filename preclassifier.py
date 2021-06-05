"""
A script to train a classifier to separate "bad images" (i.e., a human cannot
reliably identify the image as having snow or not, due to lighting conditions)
from "good images".
"""

# Standard library imports
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Local application imports
from utils.errors import eprint, warn
from utils.progress_bar import ProgressBar
from utils.validate_args import validate_csv, validate_directory


def main():
    # Parse and validate arguments
    parser = ArgumentParser(description='trains a classifier')
    parser.add_argument('meanvar', action='store',
                        help='the CSV file containing the image means and \
                            variances (see sat_meanvar.py)')
    parser.add_argument('labels', action='store',
                        help='the CSV file containing the image labels (see \
                            label_data.py)')
    args = parser.parse_args()

    meanvar_csv = validate_csv(args.meanvar, panic_on_overwrite=False)
    label_csv = validate_csv(args.labels, panic_on_overwrite=False)
    meanvar_df = pd.read_csv(meanvar_csv)
    label_df = pd.read_csv(label_csv)
    joint_df = meanvar_df.merge(label_df)

    X = joint_df[['mean', 'variance']].to_numpy()
    y = joint_df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=42)
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = f1_score(y_test, predictions)
    print(score)


if __name__ == '__main__':
    main()

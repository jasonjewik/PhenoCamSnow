"""
A script to train and save a Nu SVC to classify the image data from the
canadaobj site. Classifies each image as "bad image", "no snow" or, "has snow".
"""

# Standard library imports
from argparse import ArgumentParser
import csv
from pathlib import Path
import joblib
import sys

# Third party imports
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC


def main():
    parser = ArgumentParser(description='Trains a Nu SVC to classify images.')
    parser.add_argument('labels', action='store',
                        help='the csv file containing the image labels (must \
                            match the format returned by label_data.py')
    parser.add_argument('input_data', action='store',
                        help='the csv file containing the image features')
    parser.add_argument('-o', '--output', action='store', default='model.joblib',
                        help='the name of the output file, defaults to model.joblib')
    args = parser.parse_args()
    csv_data = dict()

    label_fp = Path(args.labels).resolve()
    ftrs_fp = Path(args.input_data).resolve()

    if not label_fp.exists():
        print(f'error: {label_fp} does not exist')
        sys.exit(1)
    if not label_fp.is_file() or label_fp.suffix != '.csv':
        print(f'error: {label_fp} is not a valid file')
        sys.exit(1)
    if not ftrs_fp.exists():
        print(f'error: {ftrs_fp} does not exist')
        sys.exit(1)
    if not ftrs_fp.is_file() or ftrs_fp.suffix != '.csv':
        print(f'error: {ftrs_fp} is not a directory')
        sys.exit(1)

    with open(label_fp) as f:
        reader = csv.reader(f)
        for fname, label in reader:
            csv_data[fname] = [int(label)]

    with open(ftrs_fp) as f:
        reader = csv.reader(f)
        for row in reader:
            fname, ftrs = parse_row(row)
            if fname in csv_data:
                csv_data[fname].append(ftrs)

    X, y = [], []
    for label, ftrs in csv_data.values():
        X.append(ftrs)
        if label == 2 or label == 3:  # merges the two snow classes
            y.append(2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = make_pipeline(
        StandardScaler(),
        NuSVC(nu=0.15, class_weight='balanced'))
    print('Started training')
    clf.fit(X_train, y_train)
    print('Done training')
    predictions = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')
    print('Class-wise F1 Scores:', end=' ')
    print(f'{f1_score(y_test, predictions, average=None)}')

    outpath = Path(args.output).resolve()
    proceed = True
    if outpath.exists() and outpath.is_file():
        print(f'warning: {outpath} already exists.')
        key = input('overwrite? [y/N] ')
        key = key.lower()
        if key == 'y':
            proceed = True
        else:
            proceed = False

    if not proceed:
        print('warning: will not check again for overwrite')
        out_name = input('provide a new name for the output file: ')
        outpath = Path(out_name).resolve()

    with open(outpath, 'wb') as f:
        joblib.dump(clf, f)
    print(f'Saved model to {outpath}')


def parse_row(row):
    fname = row[0]
    ftrs = list()
    for x in row[1:]:
        x = x.replace('[', ' ')
        x = x.replace(']', ' ')
        x = x.strip().split()
        for num in x:
            ftrs.append(float(num))
    return fname, ftrs


if __name__ == '__main__':
    main()

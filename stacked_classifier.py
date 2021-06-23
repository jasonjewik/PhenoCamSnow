"""
A script that runs the saturation and snow classifiers on the given set of
images.
"""

# Standard library imports
from argparse import ArgumentParser
import joblib
from pathlib import Path

# Local application imports
import utils
import sat_classifier
import snow_classifier


def main():
    # Parse and validate arguments
    parser = ArgumentParser(
        description='Predicts snow versus no-snow for the given images.')
    parser.add_argument('sat', metavar='saturation', help='the "bad image" \
        classifier, saved in .joblib format')
    parser.add_argument('snow', help='the snow classifier, saved in .joblib \
        format')
    parser.add_argument('-i', '--images', action='store', nargs='+',
                        default=Path.cwd(),
                        help='paths of the directories containing the images \
                            to be classified; defaults to the current working \
                                directory')
    parser.add_argument('-o', '--output', action='store',
                        default='results.csv', help='the results will be \
                            written to this csv file; defaults to \
                                "results.csv"')
    args = parser.parse_args()

    sat_model_path = utils.validate_file(
        args.sat, extension='.joblib', panic_on_overwrite=False)
    snow_model_path = utils.validate_file(
        args.snow, extension='.joblib', panic_on_overwrite=False)
    image_dirs = [utils.validate_directory(direc) for direc in args.images]
    outpath = utils.validate_file(args.output, extension='.csv')

    # Load models and make predictions
    sat_model = joblib.load(sat_model_path)
    snow_model = joblib.load(snow_model_path)

    print('Filtering "bad images"')
    sat_df = None
    for direc in image_dirs:
        temp_df = sat_classifier.predict(sat_model, direc)
        temp_df['image'] = temp_df['image'].apply(
            lambda x: Path.joinpath(direc, x))
        sat_df = temp_df if sat_df is None else sat_df.append(temp_df)

    print('Performing snow classification')
    snow_df = None
    pgbar = utils.ProgressBar(len(sat_df))
    for row in sat_df.itertuples():
        pgbar.display()
        if row.label == 1:
            temp_df = snow_classifier.predict(
                snow_model, single_image=row.image)
            snow_df = temp_df if snow_df is None else snow_df.append(temp_df)
        pgbar.inc()
        pgbar.display()
    snow_df.to_csv(outpath, index=False)


if __name__ == '__main__':
    main()

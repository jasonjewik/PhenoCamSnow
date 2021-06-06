# Snow Classifier

**Goal:** To classify images taken at the [BERMS Old Jack Pine Site, Saskatchewan, Canada](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/) (canadaojp) into "snow" versus "no snow".

**To Do:**

- [x] Modularize re-used code (especially in `run_classifier.py`)
- [x] Data pre-processing needs to crop out the ~~top part of the image (color of the sky might influence classification)~~ bottom part of the image (no need to sample the black bar)
- [x] Write another classifier to filter out "bad images" ([for example](https://phenocam.sr.unh.edu/data/archive/canadaojp/2020/11/canadaojp_2020_11_30_175959.jpg)).
- [x] Output per-image color percentages (i.e., how much of this image is the first identified color, the second, etc.?)
- [ ] ~~Maybe we can simplify the SVM inputs to just be the lightest (highest value in HSV) identified color in the image?~~
- [ ] Maybe we can also use previous classifications to influence the next? (e.g., if the previous 3 images were snow, maybe the next will also be snow)
- [ ] Auto hyperparameter tuning
- [ ] Write out errors/warnings to log files
- [ ] Figure out why snow classifier performs poorly on evaluation
- [ ] Clean conda env
- [ ] Combine all the scripts that do image feature extraction

**Known Issues:**

- The original goal was to classify into three classes: "no snow", "snow on ground", and "snow on canopy"
- Snow classifier supports evaluation on multiple image directories, but not prediction

## Usage

Before running any script in this repository, activate the conda environment:

```
conda activate snow-classifier
```

### Main model

The main model first separates out any "bad images" then runs snow classification on the rest. It has not yet been written.

### Submodels

There are two sub-models. The first one is `sat_classifier.py`, which determines whether an image is "good" or "bad" quality. In other words, could a human reliably identify this image as having snow or not? The model classifies images according to the mean and variance of their saturation (computed by converting from RGB to HSV color space). The model `sat_clf.joblib` was trained using the data in `csv/sat_clf`. It is a Linear SVM with default parameters trained with 67/33 train-test split. On the training data, the model achieves accuracy of 1.0 and F1 score of 1.0. This model's predictions on the sample images can be found in `images/sat_predictions.csv`, where 0 means "image is good/saturated" and 1 means "image is bad/undersaturated".

To run the sample model on the sample images,

```
python sat_classifier.py --predict models/sat_clf.joblib --images images --output images/sat_predictions.csv
```

The second one is `snow_classifier.py`, which determines whether snow is present in an image. This assumes no input images are "bad". The model `snow_clf.joblib` was trained using the data in `csv/snow_clf` (except for `eval_labels.csv`, which are reserved for evaluation). It is a Nu SVM with parameters nu = 0.1 and class weights = balanced with 67/33 train-test split. On the training data, the model acheives accuracy of 0.9894 and F1 score of 0.9900. On the evaluation data, the model achieves accuracy of 0.9269 and F1 score of 0.9214. This is quite poor, so the differences have been printed to `csv/snow_clf/diff2.csv` for further investigation later. (The file `diff.csv` was for a model without standard scaling.) The model's predictions on the sample images can be found in `images/snow_predictions.csv`, where 0 means "no snow" and 1 means "has snow".

To run the sample model on the sample images,

```
python snow_classifier.py --predict models/snow_clf.joblib --output images/snow_predictions.csv --images images
```

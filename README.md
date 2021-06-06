# Snow Classifier

**Goal:** To classify images taken at the [BERMS Old Jack Pine Site, Saskatchewan, Canada](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/) (canadaojp) into "snow" versus "no snow".

**To Do:**

- [ ] Modularize re-used code (especially in `run_classifier.py`)
- [ ] Data pre-processing needs to crop out the top part of the image (color of the sky might influence classification)
- [x] Write another classifier to filter out "bad images" ([for example](https://phenocam.sr.unh.edu/data/archive/canadaojp/2020/11/canadaojp_2020_11_30_175959.jpg)).
- [ ] Output per-image color percentages (i.e., how much of this image is the first identified color, the second, etc.?)
- [ ] Maybe we can simplify the SVM inputs to just be the lightest (highest value in HSV) identified color in the image?
- [ ] Maybe we can also use previous classifications to influence the next? (e.g., if the previous 3 images were snow, maybe the next will also be snow)
- [ ] Auto hyperparameter tuning

**Known Issues:**

- The original goal was to classify into three classes: "no snow", "snow on ground", and "snow on canopy"
- ~~The model does not account for pitch black images, which should be discarded~~

## Usage

There are two models. The first one is `sat_classifier.py`, which determines whether an image is "good" or "bad" quality. In other words, could a human reliably identify this image as having snow or not? The model classifies images according to the mean and variance of their saturation (computed by converting from RGB to HSV color space). The model `sat_clf.joblib` was trained using the data in `csv/sat_clf`. This model's predictions on the sample images can be found in `images/sat_predictions.csv`, where 0 means "image is good/saturated" and 1 means "image is bad/undersaturated".

To run the sample model on the sample images,

```
conda activate snow-classifier
python sat_classifier.py --predict models/sat_clf.joblib --images images --output images/sat_predictions.csv
```

See `sat_classifier.py` for more info about how to use this model.

## Training Pipeline

Training data for the sample model consists of RGB images taken at the canadaojp site between 08:00-20:00 local time every day of July 2016, November 2016, and January 2017 (excluding 1/11-1/14, for which no image data is available). It also includes all images for each day of Janurary 2016 and May 2016.

1. Images are labeled as one of the following options using the data labeling tool
   - 0 = bad image
   - 1 = no snow
   - 2 = snow on ground
   - 3 = snow on canopy
2. Images are quantized with K-Means to reduce the number of colors.
3. A Nu SVC is trained using 67% of the labeled data and evaluated for accuracy and F1 score on the remaining 33%.

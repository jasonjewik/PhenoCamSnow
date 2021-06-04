# Snow Classifier

**Goal:** To classify images taken at the [BERMS Old Jack Pine Site, Saskatchewan, Canada](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/) (canadaojp) into "snow" versus "no snow".

**To Do:**

- [ ] Modularize re-used code (especially in `run_classifier.py`)
- [ ] Data pre-processing needs to crop out the top part of the image (color of the sky might influence classification)
- [ ] Write another classifier to filter out "bad images" ([for example](https://phenocam.sr.unh.edu/data/archive/canadaojp/2020/11/canadaojp_2020_11_30_175959.jpg)).
- [ ] Output per-image color percentages (i.e., how much of this image is the first identified color, the second, etc.?)
- [ ] Maybe we can simplify the SVM inputs to just be the lightest (highest value in HSV) identified color in the image?
- [ ] Maybe we can also use previous classifications to influence the next? (e.g., if the previous 3 images were snow, maybe the next will also be snow)
- [ ] Auto hyperparameter tuning

**Known Issues:**

- The original goal was to classify into three classes: "no snow", "snow on ground", and "snow on canopy"
- The model does not account for pitch black images, which should be discarded

## Usage

To run the sample model on data

```
conda activate snow-classifier
python run_classifier.py models/model.joblib images
```

The results of the above code can be found in `images/results.csv`.

- The first column is the file name.
- The second column is the timestamp.
- The third column is the predicted label, where
  - 0 means "no snow"
  - 1 means "has snow"

Using the sample data, we can re-train the classifier.

```
python train_classifier.py csv/sample_labels.csv csv/sample_clusters.csv
```

## Training Pipeline

Training data for the sample model consists of RGB images taken at the canadaojp site between 08:00-20:00 local time every day of July 2016, October 2016, and January 2017 (excluding 1/11-1/14, for which no image data is available). It also includes all images for each day of Janurary 2016 and May 2016.

1. Images are labeled as one of the following options using the data labeling tool
   - 0 = bad image
   - 1 = no snow
   - 2 = snow on ground
   - 3 = snow on canopy
2. Images are quantized with K-Means to reduce the number of colors.
3. A Nu SVC is trained using 67% of the labeled data and evaluated for accuracy and F1 score on the remaining 33%.

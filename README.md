# Snow Classifier

**Goal:** To classify images taken at the [BERMS Old Jack Pine Site, Saskatchewan, Canada](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/) (canadaojp) into "snow" versus "no snow".

**To Do:**

- [ ] Modularize re-used code (especially in `run_classifier.py`)

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

Training data consists of RGB images taken at the canadaojp site between 08:00-20:00 local time every day of July 2016, October 2016, and January 2017 (excluding 1/11-1/14, for which no image data is available).

1. Images are labeled as one of the following options using the data labeling tool
   - 0 = no snow
   - 1 = snow on ground
   - 2 = snow on canopy
2. Images are quantized with K-Means to reduce the number of colors.
3. A Nu SVC is trained using 67% of the labeled data and evaluated for accuracy and F1 score on the remaining 33%.

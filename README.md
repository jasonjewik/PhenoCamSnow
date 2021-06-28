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
- [x] ~~Figure out why snow classifier performs poorly on evaluation~~ It does well enough
- [ ] Clean conda env
- [ ] ~~Combine all the scripts that do image feature extraction~~
- [x] Remove the option to select the number of image clusters
- [x] Clean data labeling code since it now allows for checking of labels too

**Known Issues:**

- The original goal was to classify into three classes: "no snow", "snow on ground", and "snow on canopy"

## Usage

The following lines will run the sample classifier on the sample images.

```
$ conda activate snow-classifier
$ python snow_classifier.py --predict ./models/clf.joblib -o ./images/results.csv --images ./images
```

Try `python snow_classifier.py --help` for further usage instructions.

## Sample Classifier Metrics

```
$ python snow_classifier.py --eval ./models/clf.joblib --labels ./csv/eval_labels.csv --features ./csv/img_features.csv
Accuracy: 0.994413407821229
F1 Score: [1.         0.99346405 0.97435897]
```

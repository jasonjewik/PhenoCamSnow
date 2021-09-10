# Automated PhenoCam Snow Flagging with Deep Learning

Code implementation of the model described in these [presentation slides](https://docs.google.com/presentation/d/1zFCDnZnycpJXPcuW35efhbMwoQzJRTzgZlB8ETyN7XI/edit?usp=sharing).

Relevant links:

- [The PhenoCam Network](https://phenocam.sr.unh.edu/)
- [Conda](https://conda.io/projects/conda/en/latest/index.html)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Download

```
git clone https://github.com/jasonjewik/snow-classifier
```

## Requirements

This code was developed on Windows OS. The exact specifications for the Anaconda environment used in development can be found in `windows_env.yml`.

Installation on Windows:

1. Create the environment from the environment file:
   ```
   conda env create -f windows_env.yml
   ```
2. Activate the environment:
   ```
   conda activate snow-classifier
   ```

Also provided is a manually created environment file with just the explicitly installed packages.

## Usage

### Notebook

Follow the instructions in the notebook. Code is nearly identical to the scripts, but with the additional option of labeling images in notebook.

### Script

The following code snippets assume the desired site is [canadaojp](https://phenocam.sr.unh.edu/webcam/sites/canadaojp/). If you want to train a model for another site, replace `canadaojp` with that site's canonical name, as listed on the [site table](https://phenocam.sr.unh.edu/webcam/network/table/). E.g., `canadaOBS`.

**To train a model with new data:**

```
python train.py canadaojp --new --n_train 120 --n_test 30 --categories too_dark no_snow snow
```

- Categories are separated by spaces, so any spaces in a category's name must be replaced with another character or ommitted. E.g., `too_dark` and `TooDark` are both acceptable alternatives to `too dark`.
- The path to the best model will be printed to the console and written to `best_model_paths.csv`.

**To train a model with already downloaded and labeled data:**

```
python train.py --existing --train_dir canadaojp_train --test_dir --canadaojp_test
```

- Train image annotations must be in the file `canadaojp_train/annotations.csv`, of the format returned by `utils.label_images_via_subdir`.
- Likewise for test image annotations.
- The path to the best model will be printed to the console and written to `best_model_paths.csv`.

**To get predicted categories for a local directory of images:**

```
python predict.py canadaojp path/to/best_model.pth --categories too_dark no_snow snow --directory canadaojp_test_images
```

- Replace `path/to/best_model.pth` with a path returned by `train.py`
- The order of the categories matter! Ensure these match the order in the annotations file.

**To get the predicted category for a single online image:**

```
python predict.py canadaojp path/to/best_model.pth --categories too_dark no_snow snow --url https://phenocam.sr.unh.edu/data/latest/canadaojp.jpg
```

- Same as above code snippet's notes.

## Experimental results / Examples

See the notebook `PhenoCamResNet.ipynb`.

## Acknowledgements

[Professor Jochen Stutz](https://atmos.ucla.edu/people/faculty/jochen-stutz) and [Zoe Pierrat](https://atmos.ucla.edu/people/graduate-student/zoe-pierrat).

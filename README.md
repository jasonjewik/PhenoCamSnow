<h1 align="center">PhenoCamSnow</h1>

[![Documentation Status](https://readthedocs.org/projects/phenocamsnow/badge/?version=latest)](https://phenocamsnow.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PhenoCamSnow** is a Python package for quickly building deep learning models to classify [PhenoCam images](https://phenocam.sr.unh.edu/).

## Installation

PhenoCamSnow supports Python 3.7+ and can be installed via pip:

```console
pip install phenocam-snow
```

Optional dependencies for development and documentation purposes can be installed by specifying the extras `[dev]` and `[docs]`, repsectively. 

## Example Usage

The following code snippets show how to train and evaluate a model on classifying images from the canadaojp site into "snow", "no snow", and "too dark".

```console
python -m phenocam_snow.train \
   canadaojp \
   --model resnet18 \
   --learning_rate 5e-4 \
   --weight_decay 0.01 \
   --new \
   --n_train 120 \
   --n_test 30 \
   --classes snow no_snow too_dark
```
This will print out the file path of the best model, which can be substituted into the next command.

```console
python -m phenocam_snow.predict \
   canadaojp \
   [path/to/best_model.ckpt] \
   resnet18 \
   --categories snow no_snow too_dark
   --url https://phenocam.nau.edu/data/latest/canadaojp.jpg
```

Advanced usage details can be found in the [documentation](http://phenocamsnow.readthedocs.io/).

## Citation

If you use PhenoCamSnow for your work, please see [`CITATION.cff`](CITATION.cff) or use the citation prompt provided by GitHub in the sidebar.

## Acknowledgements

[Professor Jochen Stutz](https://atmos.ucla.edu/people/faculty/jochen-stutz) and [Zoe Pierrat](https://atmos.ucla.edu/people/graduate-student/zoe-pierrat).

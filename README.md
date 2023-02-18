<h1 align="center">PhenoCamSnow</h1>

[![Documentation Status](https://readthedocs.org/projects/phenocamsnow/badge/?version=latest)](https://phenocamsnow.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PhenoCamSnow** is a Python package for quickly building deep learning models to classify [PhenoCam images](https://phenocam.sr.unh.edu/) into "has snow", "doesn't have snow", or "is too dark".

## Installation

PhenoCamSnow supports Python 3.7+ and can be installed via pip:

```console
pip install phenocam-snow
```

Optional dependencies for development and documentation purposes can be installed by specifying the extras `[dev]` and `[docs]`, repsectively. 

## Quickstart

In the following code snippets, `[site]` is a stand-in for some canonical site name as listed on [the PhenoCam website](https://phenocam.nau.edu/webcam/network/table/).

### Training a model

With new data:
```console
python train.py [site] \
   --new \
   --n_train 120 \
   --n_test 30 \
```

With already downloaded and labeled data:
```console
python train.py \
   --existing \
   --train_dir [site]_train \
   --test_dir [site]_test
```

### Getting predictions

For a local directory of images:
```console
python predict.py [site] \
   [path/to/checkpoint_of_best_model.pth] \
   --directory [site]_test_images
```

For a single online image:
```console
python predict.py [site] \
   [path/to/checkpoint_of_best_model.pth] \
   --url https://phenocam.sr.unh.edu/[path/to/image]
```


Further usage details can be found in the [documentation](http://phenocamsnow.readthedocs.io/).

## Citation

If you use PhenoCamSnow for your work, please see [`CITATION.cff`](CITATION.cff) or use the citation prompt provided by GitHub in the sidebar.

## Acknowledgements

[Professor Jochen Stutz](https://atmos.ucla.edu/people/faculty/jochen-stutz) and [Zoe Pierrat](https://atmos.ucla.edu/people/graduate-student/zoe-pierrat).

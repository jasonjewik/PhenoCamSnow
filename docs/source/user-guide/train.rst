Training a Model
================

To train a model to distinguish "snow" from "no snow" from "too dark" for the
"canadaojp" site, use the following command. For a different site, use the
canonical site name as listed at this
`table <https://phenocam.nau.edu/webcam/network/table/>`_. For different
categories, change the class names. Note that the provided names cannot have
spaces: use underscores instead.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.train \
        canadaojp \
        --model resnet18 \
        --learning_rate 5e-4 \
        --weight_decay 0.01 \
        --new \
        --n_train 120 \
        --n_test 30 \
        --classes snow no_snow too_dark

When you run this command, the program will download 120 random training images
from the canadaojp archive and 30 random testing images. The training images
are the examples the model learns from, and the testing images are for
evaluating the model's performance. After each set of images is downloaded,
the program will prompt you to label the images. To do so, it will create
three sub-directories inside of the current directory, each named after one of
the classes. Move the images into their corresponding directory (e.g., all the
pictures with snow should go into the "snow" directory) to label them.

You can also change the model. Currently, PhenocamSnow only supports training 
`ResNet <https://arxiv.org/abs/1512.03385/>`_ variants: ResNet-18, ResNet-34,
ResNet-50, ResNet-101, and ResNet-152. The expected format is "resnet{number}".
The larger the number, the more expressive the model, but the slower it might
be to train.

The learning rate and weight decay arguments are hyperparameters for the model.
The "new" flag indicates that you need to download the training and testing
images. In this case, we will be getting 120 training images and 30 testing
images. Empirically, I found this to work well for canadaojp (93% accuracy
on the testing images).

When the program has finished training the model, it will print out the model's
performance on the testing images and the file path of the saved best model.
This file path will also be written to a file called ``best_model_paths.csv``,
which will look something like this:

.. code-block:: text
    :linenos:

    timestamp,site_name,best_model
    2023-05-24T15:45:59.642605,canadaojp,/home/jason/Development/test/canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt

To get the model's predictions for a set of images, see the next section of the
documentation.

If you have further questions, please raise an
`issue on the GitHub repository <https://github.com/jasonjewik/PhenoCamSnow/issues/new/choose>`_.
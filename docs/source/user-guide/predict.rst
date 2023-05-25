Classifying New Images
======================
Models produced by PhenoCamSnow can be used to generate predictions for all
images in a local directory, or to generate a prediction for a single image as
pointed to by URL. 

Local Prediction
----------------

To get predictions for a local directory of canadaojp images, use the
following command.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
        resnet18 \
        --categories snow no_snow too_dark
        --directory canadaojp_test

The file path provided in the third line is printed by the model training
script. It is also saved in a file called ``best_model_paths.csv``. Ensure that
the categories provided are the same name and in the same order as provided
during training. Similarly, ensure that the ResNet variant specified is the
same as in training.

The program will then print out its predictions to a CSV file that looks like
this:

.. code-block:: text
    :linenos:

    # Site: canadaojp
    # Categories:
    # 0. snow
    # 1. no_snow
    # 2. too_dark
    timestamp,label
    2016-04-01 17:29:59,snow
    2017-11-13 13:59:59,snow
    2019-01-24 16:59:59,snow
    2016-04-09 20:29:59,snow
    2020-02-26 09:59:59,snow
    2016-12-21 11:29:59,snow
    2017-03-19 11:59:59,snow
    2020-10-29 13:29:59,snow
    2018-03-01 19:00:00,snow
    2016-04-24 07:29:59,no_snow
    2016-05-14 04:00:00,no_snow
    2019-09-25 09:30:00,no_snow
    2018-07-20 17:00:00,no_snow
    2017-05-27 16:29:59,no_snow
    2016-08-28 20:29:59,no_snow
    2017-06-06 09:30:00,no_snow
    2017-09-02 08:29:59,no_snow
    2019-09-09 16:00:00,no_snow
    2019-01-07 01:29:59,too_dark
    2020-01-07 23:30:00,too_dark
    2017-09-05 01:59:59,too_dark
    2018-07-24 01:30:00,too_dark
    2017-11-21 18:59:59,too_dark
    2020-01-01 03:29:59,too_dark
    2019-02-01 05:29:59,too_dark
    2020-10-14 06:30:00,too_dark
    2020-05-31 02:59:59,too_dark
    2018-12-22 02:29:59,too_dark
    2019-05-01 04:30:00,too_dark
    2020-07-22 04:00:00,too_dark

Online Prediction
-----------------

PhenoCamSnow is also capable of generating a prediction for a single online
image, as pointed to by a URL. For example, using canadaojp again:

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
        resnet18 \
        --categories snow no_snow too_dark
        --url https://phenocam.nau.edu/data/latest/canadaojp.jpg

The program will print its prediction to the console. E.g., ``no_snow``.

If you have further questions, please raise an
`issue on the GitHub repository <https://github.com/jasonjewik/PhenoCamSnow/issues/new/choose>`_.
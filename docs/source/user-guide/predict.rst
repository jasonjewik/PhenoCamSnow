Classifying New Images
======================
Models produced by PhenoCamSnow can be used to generate predictions for all
images in a local directory, or to generate a prediction for a single image as
pointed to by URL. 

To get predictions for a local directory of canadaojp images, use the
following command.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        /home/jason/Development/test/canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
        --categories snow no_snow too_dark
        --directory canadaojp_test_images

The file path provided in the third line is printed by the model training
script. Ensure that the categories provided are the same name and the same
order as provided during training. The program will then write out the model
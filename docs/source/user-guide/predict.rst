Classifying New Images
======================
Models produced by PhenoCamSnow can be used to generate predictions for all
images in a local directory, or to generate a prediction for a single image as
pointed to by URL.

To get predictions for a local directory of images in Jupyter notebook.

.. nbinput:: ipython3
    :execution-count: 1

    from phenocam_snow.predict import load_model_from_file, run_model_offline

.. nbinput:: ipython3
    :execution-count: 2

    # REPLACE ME
    model_path = "path/to/checkpoint_of_best_model.pth"
    my_model = load_model_from_file(model_path)

.. nbinput:: ipython3
    :execution-count 3:

    site_name = "canadaojp"
    img_dir = "canadaojp_test_images"
    run_model_offline(my_model, site_name, img_dir)

The same thing can be done in the command line interface.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        [path/to/checkpoint_of_best_model.pth] \
        --categories snow no_snow too_dark
        --directory canadaOBS_test_images

To get a prediction for a single online image.

.. nbinput:: ipython3
    :execution-count 1:

    from phenocam_snow.predict import load_model_from_file, run_model_online

.. nbinput:: ipython3
    :execution-count: 2

    # REPLACE ME
    model_path = "path/to/checkpoint_of_best_model.pth"
    my_model = load_model_from_file(model_path)

.. nbinput:: ipython3
    :execution-count 3:

    site_name = "canadaojp"

    categories = 

    # REPLACE ME
    img_url = "url/of/image_to_classify.png"

    run_model_online(my_model, site_name, img_dir)
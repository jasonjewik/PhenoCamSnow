Training a Model
================
Currently, PhenocamSnow only supports training 
`ResNet <https://arxiv.org/abs/1512.03385/>`_ variants: ResNet-18, ResNet-34,
ResNet-50, ResNet-101, and ResNet-152.

To train a model on new data in Jupyter notebook.

.. nbinput:: ipython3
    :execution-count: 1

    from phenocam_snow.train import train_model_with_new_data

.. nbinput:: ipython3
    :execution-count: 2

    model = "resnet18"
    # we also could have used resnet34, resnet50, resnet101, or resnet152

    # optimization hyperparameters
    learning_rate = 5e-4
    weight_decay = 0.01    # regularization term

    # we can use any canonical site PhenoCam site name
    site_name = "canadojp"
    
    label_method = "in notebook"
    # we also could have used "via subdir"

    # training/testing data split
    n_train = 120
    n_test = 30

    # these classes work well for canadaojp, but you could also do
    # classes = ["foggy", "not_foggy", "too_dark"]
    classes = ["snow", "no_snow", "too_dark"]

    train_model_with_new_data(
        model,
        learning_rate,
        weight_decay,
        site_name
        n_train,
        n_test,
        classes
    )


The same thing can be done in the command line interface.

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
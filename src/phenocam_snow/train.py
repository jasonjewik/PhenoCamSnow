# Standard library
from argparse import ArgumentParser

# Local application
from .data import *
from .model import *
from .utils import *

# Third party
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torchmetrics.functional import accuracy
from torchvision.io import read_image
import torchvision.models as models
import torchvision.transforms as transforms


def main():
    parser = ArgumentParser(
        description="Train a model to classify images from a given PhenoCam site"
    )
    parser.add_argument("site_name", help="The PhenoCam site to train on.")
    parser.add_argument(
        "--new",
        action="store_true",
        default=False,
        help="If given, trains and tests on new data. \
                            --n_train, --n_test, --categories are required.",
    )
    parser.add_argument(
        "--n_train", type=int, help="The number of train images to use."
    )
    parser.add_argument("--n_test", type=int, help="The number of test images to use.")
    parser.add_argument(
        "--existing",
        action="store_true",
        default=False,
        help="If given, trains and tests on existing data. \
                            --train_dir, --test_dir are required",
    )
    parser.add_argument(
        "--train_dir", help="The file path of the train images directory."
    )
    parser.add_argument(
        "--test_dir", help="The file path of the test images directory."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="If specified, everything is seeded with 42",
    )
    args = parser.parse_args()

    if args.deterministic:
        pl.seed_everything(42, workers=True)

    if args.new and args.existing:
        print("Cannot specify both --new and --existing")
    elif args.new:
        label_method = "via subdir"  # can't do "in notebook" from a script
        train_model_with_new_data(
            args.site_name, label_method, args.n_train, args.n_test, args.deterministic
        )
    elif args.existing:
        train_model_with_existing_data(
            args.site_name, args.train_dir, args.test_dir, args.deterministic
        )
    else:
        print("Please specify either --new or --existing")


def train_model_with_new_data(site_name, label_method, n_train, n_test, deterministic):
    """Pipeline for building a model on new data.

    :param site_name: The name of the PhenoCam site you want.
    :type site_name: str
    :param label_method: How you wish to label images ("in notebook" or "via
        subdir").
    :type label_method: str
    :param n_train: The number of training images to use.
    :type n_train: int
    :param n_test: The number of testing images to use.
    :type n_test: int
    :param deterministic: Whether the trainer should have the `deterministic`
        flag enabled. False, by default.
    :type deterministic: bool
    :return: The best model obtained during training.
    :rtype: PhenoCamResNet
    """
    ##############################
    # 1. Download and label data #
    ##############################
    categories = ["too_dark", "no_snow", "snow"]
    valid_label_methods = ["in notebook", "via subdir"]
    assert label_method in valid_label_methods

    train_dir = f"{site_name}_train"
    test_dir = f"{site_name}_test"
    train_labels = f"{train_dir}/labels.csv"
    test_labels = f"{test_dir}/labels.csv"

    dm_args = dict(
        site_name=site_name,
        train_dir=train_dir,
        train_labels=train_labels,
        test_dir=test_dir,
        test_labels=test_labels,
    )
    dm = PhenoCamDataModule(**dm_args)

    # Train data arguments
    train_download_args = dict(
        site_name=site_name,
        dates=get_site_dates(site_name),
        save_to=train_dir,
        n_photos=n_train,
    )
    # Train label arguments
    train_label_args = dict(
        site_name=site_name,
        categories=categories,
        img_dir=train_dir,
        save_to=train_labels,
        method=label_method,
    )

    # Test data arguments
    test_download_args = dict(
        site_name=site_name,
        dates=get_site_dates(site_name),
        save_to=test_dir,
        n_photos=n_test,
    )
    # Test label arguments
    test_label_args = dict(
        site_name=site_name,
        categories=categories,
        img_dir=test_dir,
        save_to=test_labels,
        method=label_method,
    )

    dm.prepare_data(
        train_download_args, train_label_args, test_download_args, test_label_args
    )

    ##################
    # 2. Train model #
    ##################
    dm.setup(stage="fit")
    model = PhenoCamResNet(n_classes=len(categories))
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{site_name}_pytorch_lightning_logs")
    trainer = pl.Trainer(
        log_every_n_steps=6,
        gradient_clip_val=0.01,
        max_epochs=50,
        logger=logger,
        deterministic=deterministic,
    )
    trainer.fit(model, dm)

    #####################
    # 3. Evaluate model #
    #####################
    dm.setup(stage="test")
    trainer.test(datamodule=dm, ckpt_path="best")

    ######################
    # 4. Note best model #
    ######################
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_models_csv = Path("best_model_paths.csv")
    if not best_models_csv.exists():
        with open(best_models_csv, "w") as f:
            f.write("timestamp,site_name,best_model\n")
    with open(best_models_csv, "a") as f:
        f.write(f"{datetime.now().isoformat()},{site_name},{best_model_path}\n")

    ########################
    # 5. Return best model #
    ########################
    best_model = model.load_from_checkpoint(best_model_path)
    best_model.freeze()
    print(f"Path of best model: {best_model_path}")
    return best_model


def train_model_with_existing_data(site_name, train_dir, test_dir, deterministic):
    """Pipeline for building model with already downloaded/labeled data.

    :param site_name: The name of the PhenoCam site you want.
    :type site_name: str
    :param train_dir: Directory with training images and labels.
    :type train_dir: str
    :param test_dir: Directory with testing images and labels.
    :type test_dir: str
    :param deterministic: Whether the trainer should have the `deterministic`
    flag enabled. False, by default.
    :type deterministic: bool
    :return: The best model obtained during training.
    :rtype: PhenoCamResNet
    """

    ###################
    # 1. Prepare data #
    ###################
    if type(train_dir) is str:
        train_dir = Path(train_dir)
    if type(test_dir) is str:
        test_dir = Path(test_dir)
    dm_args = dict(
        site_name=site_name,
        train_dir=train_dir,
        train_labels=train_dir.joinpath("labels.csv"),
        test_dir=test_dir,
        test_labels=test_dir.joinpath("labels.csv"),
    )
    dm = PhenoCamDataModule(**dm_args)
    dm.prepare_data()

    ##################
    # 2. Train model #
    ##################
    dm.setup(stage="fit")
    model = PhenoCamResNet(n_classes=len(dm.get_categories()))
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{site_name}_pytorch_lightning_logs")
    trainer = pl.Trainer(
        log_every_n_steps=6,
        gradient_clip_val=0.01,
        max_epochs=50,
        logger=logger,
        deterministic=deterministic,
    )
    trainer.fit(model, dm)

    #####################
    # 3. Evaluate model #
    #####################
    dm.setup(stage="test")
    trainer.test(datamodule=dm, ckpt_path="best")

    ######################
    # 4. Note best model #
    ######################
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_models_csv = Path("best_model_paths.csv")
    if not best_models_csv.exists():
        with open(best_models_csv, "w") as f:
            f.write("timestamp,site_name,best_model\n")
    with open(best_models_csv, "a") as f:
        f.write(f"{datetime.now().isoformat()},{site_name},{best_model_path}\n")

    ########################
    # 5. Return best model #
    ########################
    best_model = model.load_from_checkpoint(best_model_path)
    best_model.freeze()
    print(f"Path of best model: {best_model_path}")
    return best_model


if __name__ == "__main__":
    main()

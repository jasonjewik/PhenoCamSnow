# Standard library imports
from typing import List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchmetrics.functional import accuracy
from torchvision.io import read_image
import torchvision.models as models
import torchvision.transforms as transforms

# Local application imports
from utils import *


class PhenocamImageDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None):
        """
        ----------
        Parameters
        ----------
        img_dir: The directory where all the images are contained.
        ann_file: The file path of the annotations file for the images in img_dir.
            See read_annotations().
        transform: The transform to apply to images.
        target_transform: The transform to apply to image labels.
        """
        df = read_annotations(ann_file)
        self.img_labels = df[['img_name', 'int_label']].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = read_image(img_path) / 255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class PhenoCamDataModule(pl.LightningDataModule):
    def __init__(self,
                 site_name: str,
                 train_dir: Union[str, Path],
                 train_anns: Union[str, Path],
                 test_dir: Union[str, Path],
                 test_anns: Union[str, Path],
                 batch_size: Optional[int] = 16) -> None:
        """
        ----------
        Parameters
        ----------
        site_name: The name of the Phenocam site this data module is for.
        train_dir: The directory containing the train images.
        train_anns: The path to the train annotations.
        test_dir: The directory containing the test images.
        test_anns: The path to the test annotations.
        batch_size: The batch size for the data module. (Default is 16).

        ----------
        Returns
        ----------
        None.
        """
        super().__init__()

        self.site_name = site_name
        self.train_dir = train_dir
        self.train_annotations = train_anns
        self.test_dir = test_dir
        self.test_annotations = test_anns
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.aug_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.RandomApply([transforms.GaussianBlur(3)]),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        # I once tried applying other augmentation steps in between
        # resizing and blurring but found that these greatly reduced
        # accuracy on the canadaOBS site - often lead to NaN loss.

        # Preprocessing steps applied to test data
        self.std_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def prepare_data(self,
                     train_download_args: Optional[dict] = None,
                     train_label_args: Optional[dict] = None,
                     test_download_args: Optional[dict] = None,
                     test_label_args: Optional[dict] = None) -> None:
        """
        ----------
        Parameters
        ----------
        train_download_args: Arguments for downloading train images. See download().
        train_label_args: Arguments for labeling train images. Needs to have a key
            called 'method' which has two valid values: 'in notebook' and 'via subdir'.
            If method is 'in_notebook', then the other keys should match the parameters
            required by label_images_in_notebook(). If the method is 'via_subdir', then
            the other keys should match the parameters required by
            label_images_via_subdir().
        test_download_args: Arguments for downloading test images. See download().
        test_label_args: Argument for labeling test images. See the note for 
            train_label_args.

        ----------
        Returns
        ----------
        None.

        ----------
        Example
        ----------
        >>> train_download_args = dict(
            site_name='vindeln2',
            dates=get_site_dates('vindeln2'),
            save_to='train_dir',
            n_photos=120)
        >>> train_label_args = dict(
            method='in notebook',
            site_name='vindeln2'
            categories=['no snow', 'too dark', 'snow']
            img_dir='train_dir'
            save_to='train_dir/annotations.csv')
        >>> test_download_args = dict(
            site_name='vindeln2', # the train and test sites must be the same
            dates=get_site_dates('vindeln2'),
            save_to='test_dir',
            n_photos=30)
        >>> test_label_args = dict(
            method='via subdir',
            site_name='vindeln2',
            categories=['no snow', 'too dark', 'snow'] # the order of the categories must be preserved
            img_dir='test_dir',
            save_to='test_dir/annotations.csv')
        >>> prepare_data(train_download_args, 
                         train_label_args,
                         test_download_args,
                         test_label_args)
        """
        # Download and/or label train data
        if train_download_args:
            assert train_download_args['site_name'] == self.site_name
            assert train_download_args['save_to'] == self.train_dir
            print('Downloading train data')
            download(**train_download_args)
        if train_label_args:
            assert train_label_args['img_dir'] == self.train_dir
            assert train_label_args['save_to'] == self.train_annotations
            print('Labeling train data')
            if train_label_args['method'] == 'via subdir':
                del train_label_args['method']
                label_images_via_subdir(**train_label_args)

        # Download and/or label test data
        if test_download_args:
            assert test_download_args['site_name'] == self.site_name
            assert test_download_args['save_to'] == self.test_dir
            print('Downloading test data')
            download(**test_download_args)
        if test_label_args:
            assert test_label_args['img_dir'] == self.test_dir
            assert test_label_args['save_to'] == self.test_annotations
            print('Labeling test data')
            if test_label_args['method'] == 'via subdir':
                del test_label_args['method']
                label_images_via_subdir(**test_label_args)

    def setup(self, stage: Optional[str] = None):
        """
        ----------
        Parameters
        ----------
        stage: If the stage is 'fit', the train data is split 70/30 into train and validation
            sets. The augmented transformation policy is applied to the images. If the stage is
            'test', the test dataset is loaded and the standard transformation is applied to
            the images. (Default is None, in which case, the train, validation, and test
            datasets are all loaded.)

        ----------
        Returns
        ----------
        None.
        """
        if stage in ('fit', None):
            img_dataset = PhenocamImageDataset(
                self.train_dir, self.train_annotations, transform=self.aug_transform)
            train_size = round(len(img_dataset) * 0.7)
            val_size = len(img_dataset) - train_size
            self.img_train, self.img_val = random_split(
                img_dataset, [train_size, val_size])
            self.dims = self.img_train[0][0].shape
        if stage in ('test', None):
            self.img_test = PhenocamImageDataset(
                self.test_dir, self.test_annotations, transform=self.std_transform)
            self.dims = getattr(self, 'dims', self.img_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.img_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.img_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def get_categories(self) -> List[str]:
        """ 
        ----------
        Returns
        ----------
        A list of the image categories, ordered according to their
        integer encoding.
        """
        train_categories = []
        with open(self.train_annotations, 'r') as f:
            start_reading = False
            for line in f:
                if start_reading:
                    if line[0] != '#':
                        break
                    else:
                        _, str_label = line[1:].split('. ')
                        str_label = str_label.strip()
                        train_categories.append(str_label)
                if line == '# Categories:\n':
                    start_reading = True

        test_categories = []
        with open(self.test_annotations, 'r') as f:
            start_reading = False
            for line in f:
                if start_reading:
                    if line[0] != '#':
                        break
                    else:
                        _, str_label = line[1:].split('. ')
                        str_label = str_label.strip()
                        test_categories.append(str_label)
                if line == '# Categories:\n':
                    start_reading = True

        assert train_categories == test_categories
        return train_categories


class PhenoCamResNet(pl.LightningModule):
    def __init__(self,
                 n_classes: int,
                 lr: Optional[float] = 2e-4) -> None:
        """
        ----------
        Parameters
        ----------
        n_classes: The number of classes to identify.
        lr: The learning rate. (Default is 2e-4, which I empirically found to produce
            good results.)

        ----------
        Returns
        ----------
        None.
        """
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr  # Leaving this here in case we want to do auto LR tuning in the future

        # Initialize a pretrained Resnet18
        backbone = models.resnet18(pretrained=True)
        n_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        # Freeze the feature extraction layers
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Use results of pre-trained feature extractor to classify
        self.classifier = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            z = self.classifier(x)
            z = F.log_softmax(z, dim=1)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        z = self.classifier(x)
        z = F.log_softmax(z, dim=1)
        loss = F.cross_entropy(z, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            z = self.classifier(x)
            z = F.log_softmax(z, dim=1)
        loss = F.cross_entropy(z, y)
        preds = torch.argmax(z, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

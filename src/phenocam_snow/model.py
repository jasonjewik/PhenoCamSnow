# Local application
from .utils import *

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics.functional import accuracy
import torchvision.models as models


class PhenoCamResNet(pl.LightningModule):
    """Loads pre-trained ResNet18 for fine-tuning."""

    def __init__(self, lr=2e-4):
        """
        :param lr: The learning rate. Default is 2e-4.
        :type lr: float
        """
        super().__init__()

        self.save_hyperparameters()
        self.lr = (
            lr  # Leaving this here in case we want to do auto LR tuning in the future
        )

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
        n_classes = 3
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
        self.log("train_loss", loss)
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
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

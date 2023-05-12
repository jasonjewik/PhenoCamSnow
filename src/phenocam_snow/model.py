# Local application
from .utils import *

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import torchvision.models as models


class PhenoCamResNet(pl.LightningModule):
    """Loads pre-trained ResNet for fine-tuning."""

    def __init__(self, resnet, n_classes, lr=5e-4, weight_decay=0.01):
        """
        :param resnet: The ResNet variant to use.
        :type resnet: str
        :param n_classes: The number of classes
        :type n_classes: int
        :param lr: The learning rate. Default is 1e-5.
        :type lr: float
        :param weight_decay: The weight decay to use. Default is 0.1.
        :type weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()
        if resnet == "resnet18":
            backbone = models.resnet18(models.ResNet18_Weights.DEFAULT)
        elif resnet == "resnet34":
            backbone = models.resnet34(models.ResNet34_Weights.DEFAULT)
        elif resnet == "resnet50":
            backbone = models.resnet50(models.ResNet50_Weights.DEFAULT)
        elif resnet == "resnet101":
            backbone = models.resnet101(models.ResNet101_Weights.DEFAULT)
        elif resnet == "resnet152":
            backbone = models.resnet152(models.ResNet152_Weights.DEFAULT)
        else:
            raise NotImplementedError(
                f"{resnet} does not exist, please choose from resnet18,"
                " resnet34, resnet50, resnet101, or resnet152"
            )
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        n_filters = backbone.fc.in_features
        self.classifier = nn.Linear(n_filters, n_classes)
        self.metric = MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        yhat = self.classifier(x)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        acc = self.metric(preds, y)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        acc = self.metric(preds, y)
        if stage:
            self.log_dict(
                {f"{stage}_loss": loss, f"{stage}_acc": acc},
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex
from seg2ml.models.lightning_model import LightningModel, Output
from seg2ml.utils.helper import create_class_instance

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetBig(LightningModel):
    def __init__(
        self,
        num_classes: int = 1,
        learning_rate: float = 0.01,
        loss_function: Dict = None,
        metrics: List[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            learning_rate_scheduler_config=kwargs.get("learning_rate_scheduler", None),
            weight_decay=kwargs.get("weight_decay", 0),
            optimizer_config=kwargs.get("optimizer", None),
            seed=kwargs.get("seed", 0)
        )

        self.num_classes = num_classes
        self.loss_function = self.init_loss_function(loss_function)
        self.segmentation_metrics = self.init_segmentation_metrics(metrics)

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs):
        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            y_target = y_target.type(torch.LongTensor).to(self.device)
        y_target = y_target.to(torch.float32)
        loss = self.loss_function(y_pred, y_target)
        return loss

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.Tensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], train=True, logging_prefix="train", sync_dist=False)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=False)
        return output

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], train=False, logging_prefix="valid", sync_dist=True)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], train=False, logging_prefix="test", sync_dist=True)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        return output

    def process_batch(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        x, y = batch[0], batch[1]
        y_pred = self.forward(x)
        loss = self.loss(y_pred=y_pred, y_target=y)
        output = {"loss": loss, "pred": y_pred.detach(), "target": y.detach()}
        return output

    def init_segmentation_metrics(self, metrics=None):
        if metrics is None:
            return [JaccardIndex()]
        return [create_class_instance(metric["module"], metric["name"], metric.get("args", {})) for metric in metrics]

    def init_loss_function(self, params: Dict = None):
        if params is None:
            return torch.nn.BCEWithLogitsLoss()
        return create_class_instance(params["module"], params["name"], params["args"])

    def log_segmentation_metrics(self, preds: List[torch.Tensor], targets: List[torch.Tensor], train: bool, logging_prefix: str, sync_dist=True):
        for metric in self.segmentation_metrics:
            metric_name = type(metric).__name__
            if isinstance(metric, SegmentationMetric):
                metric_score = metric(y_pred=preds, y_target=targets, train=train)
                self.log(f"{logging_prefix}/{metric_name}", metric_score.mean(), on_epoch=True, on_step=False, sync_dist=sync_dist)
                if self.num_classes > 1:
                    for cls_idx in range(self.num_classes):
                        self.log(f"{logging_prefix}/{metric_name}/class_{cls_idx}", metric_score[cls_idx], on_epoch=True, on_step=False, sync_dist=sync_dist)

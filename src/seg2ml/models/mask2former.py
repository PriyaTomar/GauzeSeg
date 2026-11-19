import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Resize
from torchvision.transforms import InterpolationMode
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, AutoImageProcessor
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex
from seg2ml.models.lightning_model import LightningModel, Output
from seg2ml.utils.helper import create_class_instance

logging.basicConfig()
logger = logging.getLogger(__name__)

class Mask2Former(LightningModel):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        fine_tuning: str = "model",
        model_path: str = "facebook/mask2former-swin-tiny-ade-semantic",
        learning_rate: float = 0.01,
        loss_function: Dict = None,
        metrics: List[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            learning_rate_scheduler_config=kwargs.get("learning_rate_scheduler", None),
            weight_decay=kwargs.get("weight_decay", 0),
            optimizer_config = kwargs.get("optimizer", None),
            seed=kwargs.get("seed", 0)
        )

        self.loss_function = self.init_loss_function(loss_function)
        self.segmentation_metrics = self.init_segmentation_metrics(metrics)
        self.num_classes = num_classes


        id2label = {i+1: str(i+1) for i in range(num_classes)}
        label2id = {str(i+1): i+1 for i in range(num_classes)}
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        if pretrained:
            logger.info("Initializing pre-trained model")
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                id2label=id2label,
                label2id=label2id
            )
        else:
            logger.info("Initializing model from scratch")
            config = Mask2FormerConfig.from_pretrained(
                model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                id2label=id2label,
                label2id=label2id
            )
            #may be the lcass here shoule be Mask2FormerModel
            self.model = Mask2FormerForUniversalSegmentation(config)

    def forward(self, x):
        return self.model(pixel_values=x)

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
        outputs = self.forward(x)
        class_query_logits = outputs.class_queries_logits
        mask_query_logits = outputs.masks_queries_logits
        class_probabilities = torch.nn.functional.softmax(class_query_logits, dim=-1)[..., :-1]
        mask_probabilities = torch.nn.functional.sigmoid(mask_query_logits)
        y_pred = torch.einsum("bqc, bqhw -> bchw", class_probabilities, mask_probabilities)
        y_pred = Resize(size=y.size()[2:], interpolation=InterpolationMode.NEAREST)(y_pred)
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



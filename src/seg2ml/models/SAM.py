import logging
from typing import Any, Dict, List, Tuple
import coloredlogs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex
torch.manual_seed(0)

from seg2ml.models.lightning_model import LightningModel, Output
from seg2ml.utils.helper import (
    create_class_instance,
    create_cost_function,
    create_instance,
    create_nonlinearity,
    free_params,
    frozen_params,
    get_class_nonlinearity,
    get_device,
    gumbel_softmax,
)

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class SAM(LightningModel):
    """

    Parameters
    ----------
    
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        fine_tuning: str = "model",
        model_path: str = None,
        model_name: str = None,
        nonlinearity = None,
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
        drop_out = kwargs.get("drop_out", 0)
        self.loss_function = self.init_loss_function(loss_function)
        self.segmentation_metrics = self.init_segmentation_metrics(metrics)
        self.num_classes = num_classes
        self.model = sam_model_registry[model_name](checkpoint=model_path)


    def forward(self, x):
        return self.model(x, multimask_output=True)

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs):
        """
        Calculates the loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            predicted output label by the model
        y_target: torch.Tensor
            teacher output label

        Returns
        -------
        loss value between predicted and true labels in a batch
        """
        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            y_target = y_target.type(torch.LongTensor).to(self.device)

        y_target = y_target.to(torch.float32)  # Included because of torch error
        # loss = self.loss_function(y_pred, y_target).to(torch.float32)   
        loss = self.loss_function(y_pred, y_target)

        return loss

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.Tensor], *args, **kwargs) -> Output:
        """
        Implementation of Train Step.

        Parameters
        ----------
        batch: Tuple[torch.FloatTensor, torch.Tensor]
            training batch
        args
        kwargs

        Returns
        -------
        metrics of training step
        """
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=True, logging_prefix="train")
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=False)
        return output

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Validate Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of validation step
        """
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=False, logging_prefix="valid")
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=False)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Test Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of a test step
        """
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=False, logging_prefix="test")
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=False)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Predict Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of a predict step
        """
        output = self.process_batch(batch)
        return output
    
    def predict(self, batch: torch.FloatTensor, *args, **kwargs) -> torch.Tensor:
        output = self.forward(batch)
        return output
        


    def process_batch(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        x, y = batch[0], batch[1]
        #x, y = x.to(self.device), y.to(self.device)

        y_pred = self.forward(x)
        loss = self.loss(y_pred=y_pred, y_target=y)
        #output = {"loss": loss, "pred": y_pred.detach().cpu().numpy(), "target": y.detach().cpu().numpy()}
        output = {"loss": loss, "pred": y_pred.detach(), "target": y.detach()}
        return output

    def generate(self, n: int) -> torch.tensor:
        """

        Parameters
        ----------
        n

        Returns
        -------

        """
        raise NotImplementedError("The generate method is not implemented in your class!")
    
    def init_segmentation_metrics(self, metrics = None):
        """
        Initializes the Segmentation merics.

        Parameters
        ----------
        params: Dict
            parameter to specify which metrics to use

        Returns
        -------
        List of Instances of metrics
        """

        if metrics is None:
            return JaccardIndex()
        return [create_class_instance(metric["module"], metric["name"], metric.get("args", {})) for metric in metrics]

    def init_loss_function(self, params: Dict = None):
        """
        Initializes the Loss Function.

        Parameters
        ----------
        params: Dict
            parameter to specify which loss_function to use

        Returns
        -------
        Instance of loss function
        """
        if params is None:
            return torch.nn.BCELoss()
        return create_class_instance(params["module"], params["name"], params["args"])
    
    def log_segmentation_metrics(self, preds: List[torch.Tensor], targets: List[torch.Tensor], train: bool, logging_prefix: str):
        for metric in self.segmentation_metrics:
            metric_name = type(metric).__name__
            if isinstance(metric, SegmentationMetric):
                #calculaute class-wise metric score
                metric_score = metric(y_pred= preds, y_target= targets, num_classes=self.num_classes, train=train)
                self.log(f"{logging_prefix}/{metric_name}", metric_score.mean(), on_epoch=True, on_step=False)
                for cls_idx in range(self.num_classes):
                    self.log(f"{logging_prefix}/{metric_name}/class_{cls_idx}", metric_score[cls_idx], on_epoch=True, on_step=False)


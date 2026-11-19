import warnings
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
import math
from sklearn.metrics._classification import UndefinedMetricWarning
from torchmetrics.classification import  F1Score
import torchmetrics.classification as torchmetrics_cls
from monai.metrics import SurfaceDiceMetric
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class SegmentationMetric:
    mode: str = "train"

    def __init__(self, num_classes: int = 1, evaluate_on: List[str] = ("train", "eval"), task: str = "binary"):
        self.num_classes = num_classes
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        

    def __call__(self, y_pred, y_target, train: bool = True, **kwargs) -> Optional[float]:
        self.mode = "train" if train else "eval"
        if self.mode not in self.evaluate_on:
            return None

        return self._metric(y_pred, y_target)

    def _metric(self, y_pred, y_target, num_classes: int = 2) -> Any:
        raise NotImplementedError
    
class DiceScore(SegmentationMetric):
    def __init__(self, num_classes: int = 1, evaluate_on: List[str] = ("train", "eval"), task: str = "binary", average = None):
        self.num_classes = num_classes
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        self.dice_func = F1Score(average=average, task = task, num_classes= self.num_classes)
    
    def _metric(self, pred, target):
        pred = self.activation_func(pred)
        if self.num_classes >1:
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)
        dice_func = self.dice_func.to(pred.device)
        score = dice_func(pred, target)
        return score
    
class JaccardIndex(SegmentationMetric):
    def __init__(self, num_classes: int = 1, evaluate_on: List[str] = ("train", "eval"), task: str = "binary", average = None):
        self.num_classes = num_classes
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        self.jaccard_func = torchmetrics_cls.JaccardIndex(average=average, task = task, num_classes= self.num_classes)

    def _metric(self, pred, target):
        # map logits to class labels of dim [B,C,H,W] -> [B,H,w]
        pred = self.activation_func(pred)
        if self.num_classes >1:
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)
        jaccard_func = self.jaccard_func.to(pred.device)
        score = jaccard_func(pred, target)
        # score shape for input [B,C,H,W] -> [C]
        return score

class Precision(SegmentationMetric):
    # defined only if TP+FP > 0
    def __init__(self, num_classes: int = 1, evaluate_on: List[str] = ("train", "eval"), task: str = "binary", average = None):
        self.num_classes = num_classes
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        self.precision_func = torchmetrics_cls.Precision(average=average, task = task, num_classes= self.num_classes)
        
        
    def _metric(self, pred, target):
        pred = self.activation_func(pred)
        precision_func = self.precision_func.to(pred.device)
        if self.num_classes >1:
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)
        score = precision_func(pred, target)
        return score
    
class Recall(SegmentationMetric):
    #defined when TP+FN > 0
    def __init__(self, num_classes: int = 1, evaluate_on: List[str] = ("train", "eval"), task: str = "binary", average = None):
        self.num_classes = num_classes
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        self.recall_func = torchmetrics_cls.Recall(average=average, task = task, num_classes= self.num_classes)

    def _metric(self, pred, target):
        pred = self.activation_func(pred)
        recall_func = self.recall_func.to(pred.device)
        if self.num_classes >1:
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)
        score = recall_func(pred, target)
        return score
    
class SurfaceDice(SegmentationMetric):
    #ToDO:confirm the values for multiple classes
    """
    Surface Dice metric using MONAI's implementation
    
    Args:
        num_classes (int): Number of classes
        distance_threshold (float): Distance threshold to consider surface overlap
    """
    def __init__(self, evaluate_on: List[str] = ("train", "eval"), class_thresholds: List[float] = [0.5], include_background: bool = False):
   
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid()
        self.surface_dice = SurfaceDiceMetric(
            include_background=include_background,
            class_thresholds=class_thresholds,
            distance_metric="euclidean",
            reduction="none" 
        )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Surface Dice between prediction and target.
        
        Args:
            y_pred (torch.Tensor): Prediction tensor (B, H, W)
            y_target (torch.Tensor): Target tensor (B, H, W)
            train (bool): Whether in training mode
        
        Returns:
            torch.Tensor: Surface Dice scores for each class
        """
        pred = self.activation_func(pred)
        # Convert predictions to binary
        pred = (pred > 0.5).float()
        # One-hot encode prediction and target
        # MONAI expects shape: (B, C, H, W)
        pred = torch.stack([1 - pred, pred], dim=1).squeeze(2)  # Stack background and foreground along channel dim
        target = torch.stack([1 - target, target], dim=1).squeeze(2)  # Same for target
        # # Find indices of samples that have at least one positive pixel in ground truth
         # !!!!!!!!! 
        # Filter predictions and targets to only include positive samples
        #  otherwise Monai return Nan for all 0 valu mask in ground truth
        # Get indices where target has positive values - only filter based on target
        positive_target_indices = torch.where((target[:, 1] > 0).any(dim=(1, 2)))[0]
        
        # Select samples where target has positive values, keeping all predictions
        pred = pred[positive_target_indices]
        target = target[positive_target_indices]

        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        surface_dice_score = self.surface_dice(pred, target)
        #todo: filter false negatives and if happens then set to some value, eg 0 as monai will return 
        # nan in such case
        return surface_dice_score
    
    
class HausdorffDistance(SegmentationMetric):
    def __init__(self, evaluate_on: List[str] = ("train", "eval"), class_thresholds: List[float] = [0.5]):
        self.evaluate_on: List[str] = evaluate_on
        self.activation_func = nn.Sigmoid()
        self.hausdorff_distance = torchmetrics_cls.HausdorffDistance(
            num_classes=self.num_classes,
            include_background=False,
            distance_metric="euclidean",
            reduction="none"
        )
        
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.activation_func(pred)
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        hausdorff_distance_score = self.hausdorff_distance(pred, target)
        return hausdorff_distance_score
    
    
    
    
    
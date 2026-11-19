from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch 
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import MulticlassF1Score

class SigmoidFocalLoss(torch.nn.Module):
    """Apply sigmoid focal loss https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html#sigmoid-focal-loss

    Args:
    alpha: Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default: 0.25.
    gamma: – Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: 2.
    """  
    def __init__(self, alpha: float = 0.25,
                 gamma: float = 2,
                 reduction: str = "mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args: 

        inputs: A float tensor of arbitrary shape.
        targets: A float tensor with the same shape as inputs
        """
        return sigmoid_focal_loss(inputs=inputs, targets=targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
    
# class DiceLoss(torch.nn.Module):
#     def __init__(self, num_classes: int = 2,
#                  reduction: str = "macro"):
#         super(DiceLoss, self).__init__()
#         self.num_classes = num_classes
#         self.dice_func = MulticlassF1Score(average=reduction, num_classes= self.num_classes)
        

#     def forward(self, pred, target):
#        # map logits to class labels of dim [B,C,H,W] -> [B,H,w]
#         target = torch.argmax(target, dim=1)
#         pred = torch.softmax(pred, dim=1)
#         # dice_func = Dice(average=None, num_classes= num_classes).to(pred.device)
#         dice_func = self.dice_func.to(pred.device)
#         dice_score = dice_func(pred, target)
#         loss = 1- dice_score
#         return loss

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.activation_func = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the multiclass Dice Loss.
        
        :param outputs: Model outputs (logits) of shape [B, num_classes, H, W]
        :param targets: Ground truth masks of shape [B, num_classes, H, W]
        :return: Dice loss
        """
        # Apply activation function to logits
        probs = self.activation_func(outputs)
        
        # Initialize the Dice score tensor
        dice_scores = torch.zeros(self.num_classes, device=outputs.device)
        
        for i in range(self.num_classes):
            pred = probs[:, i, :, :]
            target = targets[:, i, :, :]
            
            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target) + self.smooth
            
            dice_scores[i] = (2. * intersection + self.smooth) / union
        
        # Compute the average Dice score and return the loss
        dice_loss = 1. - torch.mean(dice_scores)
        return dice_loss


class SCELoss(torch.nn.Module):
    """ Apply symmetric cross entropy loss
    """  
    def __init__(self, alpha:float = 0.1, 
                       beta: float = 1.0):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        #q. why clip the predictions?
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        #label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        #labels are already in one_hot for due to binary masks
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(labels), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class WeightedSCELoss(torch.nn.Module):
    """ Apply symmetric cross entropy loss with weights for silver interpolations
    """  
    def __init__(self, alpha, beta, ws = 0.5, wg = 1):
        super(WeightedSCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ws = ws
        self.wg = wg
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target, label):
        # CCE
        ce_tensor = self.cross_entropy(pred, target)#B,H,W
        weight_tensor = torch.ones(size = label.size(), device=label.device)
        weight_tensor[label==1] = self.ws
        weight_tensor[label==0] = self.wg
        ce_tensor = ce_tensor.reshape(ce_tensor.shape[0],-1).mean()
        ce = (ce_tensor * weight_tensor).mean()

        # RCE
        pred = F.softmax(pred, dim=1)
        #q. why clip the predictions?
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        #label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        #labels are already in one_hot for due to binary masks
        target = torch.clamp(target, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(target), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class gce_gold_silver(nn.Module):
    def __init__(self, warmup:int  = 0, q:int = 0.1):
        """ Apply gce loss function for silver annotation

        Args:
            warmup (int, optional): warmup epochs where BCE loss is applied.. Defaults to 10.
        """        
        super(gce_gold_silver, self).__init__()
        # Check Sigmoid ?
        self.ce_function = nn.CrossEntropyLoss()
        self.warmup = warmup
        self.q = q


    def forward(self, y_pred: torch.tensor, y_true: torch.tensor, current_epoch: int, silver_list: List[int], eps=1e-12):     
        if current_epoch < self.warmup:
            loss = self.ce_function(y_pred, y_true)
            
        else:
            indices_gold = [idx for idx, value in enumerate(silver_list) if value==0]
            indices_silver = [idx for idx, value in enumerate(silver_list) if value==1]
            
            __gold_loss = self.ce_function(y_pred[indices_gold,:,:,:], y_true[indices_gold,:,:,:])
            __silver_loss = self.gce_function(y_pred[indices_silver,:,:,:], y_true[indices_silver,:,:,:], self.q)
            
            # WARNING: Do we want to implement it like that?
            
            loss = (__gold_loss + __silver_loss)/2 # denominator leads to the same influence of gold as silver even though more silver labels are present?

        return loss
    
    def gce_function(self, pred, target, q):
        loss =  target * (torch.pow(pred, q)) / q
        return loss.mean()
    
class label_smoothing(nn.Module):
    def __init__(self, warmup:int = 0, kernel_size = (3,3), sigma = (0.1, 2.0)):
        """ Apply label smooting on binary true target mask

        Args:
            warmup (int, optional): warmup epochs where BCE loss is applied.. Defaults to 10.
        """        
        super(label_smoothing, self).__init__()
        # Check Sigmoid ?
        self.ce_function = nn.CrossEntropyLoss()
        self.blur_function = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)
        self.warmup = warmup
        
    def forward(self, y_pred: torch.tensor, y_true: torch.tensor, current_epoch: int, silver_list: List[int]):     
        if current_epoch < self.warmup:
            loss = self.ce_function(y_pred, y_true)
            
        else:
            # apply label smoothinf for silver labels
            indices_silver = [idx for idx, value in enumerate(silver_list) if value==1]
            y_true[indices_silver,:,:,:] = self.blur_interpolation(y_true[indices_silver,:,:,:])
            
            # calculate loss
            loss = self.ce_function(y_true, y_pred)
        return loss
    
    def blur_interpolation(self, y: torch.tensor):
        # calculate blurring per class for binary images
        blured_samples = []
        for sample_idx in range(y.shape[0]):
            blured_list = [self.blur_function(torch.unsqueeze(y[sample_idx,:,:,i], dim= 0)) for i in range(y.shape[-1])]
            blured_array = torch.cat(blured_list,0).permute(1,2,0)
            blured_samples.append(torch.unsqueeze(blured_array, 0))
        blured_y = torch.cat(blured_samples, 0)
        return blured_y
            
        
    

if __name__ == "__main__":
    pass
    # dataset = CholecSegDataset(split="train", video_indices = [1,9,12,17,18,20,24,25,26], skip_frame_count=1)   
    # _, y_true, silver_label = dataset[-10]
    
    # kernel_size = (3,3)
    # blur_function = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    # plt.imsave("/data/Helen/surgery_video/test.png",y_true[:,:,7])
    # #test = torch.unsqueeze(y_true[:,:,7], dim =0)
    # test = torch.swapaxes(y_true, 2,0)
    # blurred = blur_function(test)
    # plt.imsave("/data/Helen/surgery_video/test_blurred.png", blurred[7,:,:])
    
    
    # y_true = torch.unsqueeze(y_true, dim=0)
    # y_true = torch.cat((y_true, y_true), 0)
    # y_pred = y_true*0.5
    # loss = label_smoothing(kernel_size=(101,101))
    # temp = loss(y_pred, y_true, 1,  [1,0])
    # define dummy Variables
    

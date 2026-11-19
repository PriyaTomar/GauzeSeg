import logging
from typing import Any, Dict, List, Tuple
import coloredlogs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex

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

class AttentionUNet(LightningModel):

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        fine_tuning: str = "model",
        model_path: str = None,
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
        self.bilinear = True
        self.in_conv = UNetConvBlock(3 , 64)
        self.Down1 = Down(64 , 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)

        self.Up1 = AttentionUp(in_channels=512+512, out_channels=256, attention_channels=256, bilinear=self.bilinear)
        self.Up2 = AttentionUp(in_channels=256+256, out_channels=128, attention_channels=128, bilinear=self.bilinear)
        self.Up3 = AttentionUp(in_channels=128+128, out_channels=64, attention_channels=64, bilinear=self.bilinear)
        self.Up4 = AttentionUp(in_channels=64+64, out_channels=64, attention_channels=32, bilinear=self.bilinear)        
        self.out_conv = OutConv(64, num_classes)

        

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)

        # Forward pass through the decoder
        x = self.Up1(x5,x4)
        x = self.Up2(x ,x3)
        x = self.Up3(x ,x2)
        x = self.Up4(x ,x1)
        
        # Final layer
        output = self.out_conv(x)
        return output


   

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
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=True, logging_prefix="train", sync_dist=False)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=False)
        # Print or log to confirm the model was loaded from a checkpoint
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
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=False, logging_prefix="valid", sync_dist=True)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
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
        self.log_segmentation_metrics(preds= output["pred"], targets= output["target"], train=False, logging_prefix="test", sync_dist=True)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
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
    
    def on_load_checkpoint(self, checkpoint):
        """This method is triggered when the model is loaded from a checkpoint."""
        print("Checkpoint has been loaded!")
        self.is_checkpoint_loaded = True

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
    
    def log_segmentation_metrics(self, preds: List[torch.Tensor], targets: List[torch.Tensor], train: bool, logging_prefix: str, sync_dist=True):
        for metric in self.segmentation_metrics:
            metric_name = type(metric).__name__
            if isinstance(metric, SegmentationMetric):
                #calculaute class-wise metric score
                metric_score = metric(y_pred= preds, y_target= targets, train=train)
                self.log(f"{logging_prefix}/{metric_name}", metric_score.mean(), on_epoch=True, on_step=False, sync_dist=sync_dist)
                if self.num_classes > 1:
                    for cls_idx in range(self.num_classes):
                        self.log(f"{logging_prefix}/{metric_name}/class_{cls_idx}", metric_score[cls_idx], on_epoch=True, on_step=False, sync_dist=sync_dist)

    

'''
Att-Unet Utilities
'''

class UNetConvBlock(nn.Module):
    " [conv -> BN -> ReLU] -> [conv -> BN -> ReLU]"
    def __init__(self, in_channels, out_channels, kernel_size = 3 , padding = True, dropout_prob = 0):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
            The Original paper uses VALID padding (i.e. no padding)
            The main benefit of using SAME padding is that the output feature map will have the same spatial dimensions
            as the input feature map.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            # Usually Conv -> BatchNormalization -> Activation
            nn.Conv2d(in_channels , out_channels , kernel_size= kernel_size , padding = int(padding)),
            nn.BatchNorm2d(out_channels), # bathc normalization necessary?
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob))

    def forward(self,inp):
        return self.double_conv(inp)


class Down(nn.Module):
    """
    Downscaling with maxpool and then Double Conv
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU
        - MaxPooling
    """
    def __init__(self, in_channels , out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate for U-Net.
        Args:
            F_g: Channels of the gating signal (upsampled decoder tensor, x1)
            F_l: Channels of the skip connection (encoder tensor, x2)
            F_int: Intermediate channels for attention mechanism
        """
        super(AttentionBlock, self).__init__()
        
        # Reduce channel dimensions with 1x1 conv for the gating signal (upsampled tensor x1)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Reduce channel dimensions with 1x1 conv for the skip connection tensor (encoder tensor x2)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Combine the gated and skip connection tensors and apply a ReLU
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal from upsampled tensor (x1)
            x: skip connection tensor (x2)
        
        Returns:
            Attention-applied skip connection
        """
        # Apply 1x1 convolution to reduce channels of gating signal
        g1 = self.W_g(g)
        # Apply 1x1 convolution to reduce channels of skip connection
        x1 = self.W_x(x)
        # Sum the transformed gating and skip connection feature maps
        psi = self.relu(g1 + x1)
        # Apply the psi function to compute attention map
        attention_map = self.psi(psi)
        # Element-wise multiply attention map with the original skip connection tensor
        out = x * attention_map
        return out

    
class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels, attention_channels, bilinear=False):
        super(AttentionUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = UNetConvBlock(in_channels, out_channels)
        # self.conv = UNetConvBlock(in_channels, out_channels)
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=attention_channels)

    def forward(self, x1, x2):
        # x1: upsampled decoder feature map
        # x2: corresponding encoder feature map (skip connection)
        
        # Upsample x1
        x1 = self.up(x1)
        #add the padding
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1 , [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        # Apply attention to x2 (skip connection)
        x2 = self.attention(x1, x2)

        # Concatenate attention-modified skip connection and upsampled tensor
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Up(nn.Module):
    """
        - Upsampling Convolution ("Up-convolution")
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU
        Upsampling vs Transposed convolutions
        Transposed convolution (a.k.a."up-convolution or fractionally-strided convolutions or deconvolutions")
            - The original paper uses this
            - detects more fine-grained detail
        Other implementation use bilinear upsampling, possibly followed by a 1x1 convolution.
        The benefit of using upsampling is that it has no parameters and if you include the 1x1 convolution,
        it will still have less parameters than the transposed convolution
    """
    def __init__(self,in_channels , out_channels , bilinear = False):
        super(Up,self).__init__()

        if bilinear: # use the normal conv to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode= 'bilinear', align_corners = True)
        else: # use Transpose convolution (the one that official UNet used)
            self.up = nn.ConvTranspose2d(in_channels//2 , in_channels // 2, kernel_size = 2,stride=2 )

        self.conv = UNetConvBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        # input dim is CHW
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1 , [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1] , dim = 1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        #return torch.softmax(self.conv(x),dim = 1)

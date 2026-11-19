import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from seg2ml.models.lightning_model import LightningModel, Output
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex
from seg2ml.utils.helper import create_class_instance

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class TransUNet(LightningModel):
    """
    Optimized TransUNet with reduced parameters (~13M)
    
    Parameters
    ----------
    num_classes: int
        Number of output classes
    input_channels: int
        Number of input channels
    img_size: int
        Size of the input image
    embedding_dim: int
        Dimension of the embedding
    num_heads: int
        Number of attention heads
    num_layers: int
        Number of transformer blocks
    learning_rate: float
        learning rate for optimizer
    **kwargs:
        additional arguments for model initialization
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        img_size: int = 512,
        embedding_dim: int = 384,  
        num_heads: int = 6,  
        num_layers: int = 8,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        lightning_params = {
            'learning_rate': learning_rate,
            'learning_rate_scheduler_config': kwargs.get('learning_rate_scheduler', None),
            'weight_decay': kwargs.get('weight_decay', 0),
            'optimizer_config': kwargs.get('optimizer', None),
            'seed': kwargs.get('seed', 0)
        }
        super().__init__(**lightning_params)
        
        self.save_hyperparameters()
        self.loss_function = self.init_loss_function(kwargs.get('loss_function', None))
        self.segmentation_metrics = self.init_segmentation_metrics(kwargs.get('metrics', None))
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Encoder channels
        self.encoder_channels = [32, 64, 128, 256]
        self.encoder = nn.ModuleList([
            self._make_encoder_block(input_channels, self.encoder_channels[0]),
            self._make_encoder_block(self.encoder_channels[0], self.encoder_channels[1]),
            self._make_encoder_block(self.encoder_channels[1], self.encoder_channels[2]),
            self._make_encoder_block(self.encoder_channels[2], self.encoder_channels[3])
        ])
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Position embedding
        self.num_patches = (img_size // 16) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        
        # Transformer
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=2.0
            ) for _ in range(num_layers)
        ])
        
        # Decoder channels
        self.decoder = nn.ModuleList([
            DecoderBlock(
                in_channels=embedding_dim,
                skip_channels=self.encoder_channels[3],
                out_channels=128
            ),
            DecoderBlock(
                in_channels=128,
                skip_channels=self.encoder_channels[2],
                out_channels=64
            ),
            DecoderBlock(
                in_channels=64,
                skip_channels=self.encoder_channels[1],
                out_channels=32
            ),
            DecoderBlock(
                in_channels=32,
                skip_channels=self.encoder_channels[0],
                out_channels=16
            )
        ])
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        input_size = x.shape[-1]  # Store original input size
        
        # Encoder
        skips = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skips.append(x)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Transformer
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        x = self.transformer(x)
        
        # Reshape back
        H = W = self.img_size // 16
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        # Decoder
        skips = skips[::-1]
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, skips[i])
        
        x = self.final_conv(x)
        
        # Add final upsampling to match input size
        if x.shape[-1] != input_size:
            x = F.interpolate(x, size=(input_size, input_size), 
                            mode='bilinear', align_corners=True)
        
        return x

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs):
        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            y_target = y_target.type(torch.LongTensor).to(self.device)
        y_target = y_target.to(torch.float32)
        return self.loss_function(y_pred, y_target)

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.Tensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], 
                                    train=True, logging_prefix="train", sync_dist=False)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=False)
        return output

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], 
                                    train=False, logging_prefix="valid", sync_dist=True)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch)
        self.log_segmentation_metrics(preds=output["pred"], targets=output["target"], 
                                    train=False, logging_prefix="test", sync_dist=True)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=False, sync_dist=True)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        return self.process_batch(batch)

    def predict(self, batch: torch.FloatTensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(batch)

    def process_batch(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        x, y = batch[0], batch[1]
        y_pred = self.forward(x)
        
        # Ensure prediction and target have the same size
        if y_pred.shape[-1] != y.shape[-1]:
            y_pred = F.interpolate(y_pred, size=y.shape[-2:], 
                                 mode='bilinear', align_corners=True)
        
        loss = self.loss(y_pred=y_pred, y_target=y)
        output = {"loss": loss, "pred": y_pred.detach(), "target": y.detach()}
        return output

    def on_load_checkpoint(self, checkpoint):
        print("Checkpoint has been loaded!")
        self.is_checkpoint_loaded = True

    def init_segmentation_metrics(self, metrics=None):
        if metrics is None:
            return JaccardIndex()
        return [create_class_instance(metric["module"], metric["name"], 
                                    metric.get("args", {})) for metric in metrics]

    def init_loss_function(self, params: Dict = None):
        if params is None:
            return torch.nn.BCELoss()
        return create_class_instance(params["module"], params["name"], params["args"])

    def log_segmentation_metrics(self, preds: List[torch.Tensor], targets: List[torch.Tensor], 
                               train: bool, logging_prefix: str, sync_dist=True):
        for metric in self.segmentation_metrics:
            metric_name = type(metric).__name__
            if isinstance(metric, SegmentationMetric):
                metric_score = metric(y_pred=preds, y_target=targets, train=train)
                self.log(f"{logging_prefix}/{metric_name}", metric_score.mean(), 
                        on_epoch=True, on_step=False, sync_dist=sync_dist)
                if self.num_classes > 1:
                    for cls_idx in range(self.num_classes):
                        self.log(f"{logging_prefix}/{metric_name}/class_{cls_idx}", 
                               metric_score[cls_idx], on_epoch=True, on_step=False, 
                               sync_dist=sync_dist)



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, 
                                        bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.attn(*[self.norm1(x)]*3)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                    kernel_size=2, stride=2)
        
        # After concatenation, input channels will be out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# Add this function to check parameters
def print_model_params(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name}: {param_count:,}")
    print(f"\nTotal parameters: {total_params:,}")
    return total_params
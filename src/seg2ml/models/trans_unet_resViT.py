import torch
import logging
from typing import Any, Dict, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import SegformerModel, SegformerConfig
from transformers import ViTModel, ViTConfig
from seg2ml.models.lightning_model import LightningModel, Output
from seg2ml.metrics.segmentation_metrics import SegmentationMetric, JaccardIndex
from seg2ml.utils.helper import create_class_instance

class TransUNetResViT(LightningModel):
    """
    TransUNet with ResNet50 backbone encoder
    Parameters
    ----------
    num_classes: int
        Number of output classes
    num_heads: int
        Number of attention heads
    num_layers: int
        Number of transformer blocks
    learning_rate: float
        learning rate for optimizer
    pretrained_resnet: bool
        Whether to load pretrained weights for ResNet50
    pretrained_vit: bool
        Whether to load pretrained weights for ViT
    model_path: str
        Path to the SegFormer model to use
    **kwargs:
        additional arguments for model initialization
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        model_path: str = "google/vit-small-patch16-224",
        **kwargs
    ):
        lightning_params = {
            'learning_rate': kwargs.get('learning_rate', 1e-4),
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
        self.drop_out = drop_out
        self.pretrained = pretrained
        
        print(f"Initializing model with pretrained={pretrained}")  # Debug print
        
        # ResNet encoder (4 layers)
        self._init_resnet_encoder(self.pretrained)
        
        # ViT transformer
        self._init_vit(self.pretrained, model_path)
        
        # Decoder with skip connections
        self._init_decoder()

    def _init_resnet_encoder(self, pretrained: bool):
        """Initialize ResNet encoder with 4 layers"""
        resnet = resnet50(pretrained=pretrained)
        
        # First conv block
        self.first_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        
        # Encoder blocks
        self.encoder = nn.ModuleList([
            resnet.layer1,  # First layer
            resnet.layer2,  # Second layer
            resnet.layer3,  # Third layer
        ])
        
        # Extract channel dimensions from ResNet
        self.encoder_channels = [64]  # First conv output channels
        for layer in self.encoder:
            # Get the output channels of each layer
            self.encoder_channels.append(layer[-1].conv3.out_channels)
        
        print(f"Encoder channels: {self.encoder_channels}")

    def _init_vit(self, pretrained: bool, model_path: str):
        """Initialize ViT transformer"""
        if pretrained:
            config = ViTConfig(
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                intermediate_size=1536,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                image_size=224,
                patch_size=16,
                num_channels=3
            )
            self.vit = ViTModel.from_pretrained(
                model_path,
                config=config
            )
        else:
            config = ViTConfig.from_pretrained(model_path)
            self.vit = ViTModel(config)
        
        # Print ViT configuration
        print("\nViT-Small Configuration:")
        print(f"Model type: {self.vit.config.model_type}")
        print(f"Hidden size: {self.vit.config.hidden_size}")
        print(f"Num attention heads: {self.vit.config.num_attention_heads}")
        print(f"Intermediate size: {self.vit.config.intermediate_size}")
        print(f"Hidden dropout: {self.vit.config.hidden_dropout_prob}")
        print(f"Attention dropout: {self.vit.config.attention_probs_dropout_prob}")
        print(f"Initializer range: {self.vit.config.initializer_range}")
        print(f"Layer norm eps: {self.vit.config.layer_norm_eps}")
        print(f"Image size: {self.vit.config.image_size}")
        print(f"Patch size: {self.vit.config.patch_size}")
        print(f"Num channels: {self.vit.config.num_channels}")
        
        # Projection layer to match ViT's input dimension
        self.projection = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], self.vit.config.hidden_size, 1),
            nn.BatchNorm2d(self.vit.config.hidden_size),
            nn.ReLU(inplace=True)
        )

    def _init_decoder(self):
        """Initialize decoder blocks using encoder channels"""
        # Use encoder channels in reverse order for decoder
        decoder_channels = self.encoder_channels[::-1]
        
        print(f"Decoder channels: {decoder_channels}")
        
        self.decoder = nn.ModuleList([
            DecoderBlock(
                in_channels=decoder_channels[i],
                skip_channels=decoder_channels[i+1] if i < len(decoder_channels)-1 else 0,
                out_channels=decoder_channels[i],
                drop_out=self.drop_out
            ) for i in range(len(decoder_channels))
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], self.num_classes, kernel_size=1),
            nn.Dropout2d(self.drop_out)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        print(f"\nInput shape: {x.shape}")
        
        # ResNet encoder path
        x = self.first_conv(x)
        x = self.maxpool(x)
        print(f"After first conv: {x.shape}")
        
        # Store skip connections
        skips = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            skips.append(x)
            print(f"Encoder block {i} output shape: {x.shape}")
        
        # Process through ViT
        x = self.projection(x)  # Project to ViT's input dimension
        print(f"After projection to ViT: {x.shape}")
        
        # Reshape for transformer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add positional embeddings
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Process through ViT
        vit_output = self.vit(x, output_hidden_states=True)
        x = vit_output.last_hidden_state
        print(f"After ViT: {x.shape}")
        
        # Reshape back to spatial format
        B, L, C = x.shape
        H = W = int((L) ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        print(f"After reshaping: {x.shape}")
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            print(f"\nDecoder block {i}:")
            print(f"Input shape: {x.shape}")
            print(f"Skip connection shape: {skips[-(i+1)].shape}")
            x = decoder_block(x, skips[-(i+1)])
            print(f"Output shape: {x.shape}")
        
        x = self.final_conv(x)
        print(f"Final output shape: {x.shape}")
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
        
        if y_pred.shape[-1] != y.shape[-1]:
            y_pred = F.interpolate(y_pred, size=y.shape[-2:],
                                 mode='bilinear', align_corners=True)
        
        loss = self.loss(y_pred=y_pred, y_target=y)
        output = {"loss": loss, "pred": y_pred.detach(), "target": y.detach()}
        return output

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, drop_out=0.1):
        super().__init__()
        # Upsampling layer - maintain input channels
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,  # Keep original input channels
            out_channels=in_channels,  # Maintain same number of channels after upsampling
            kernel_size=2,
            stride=2
        )
        
        # First conv reduces channels if needed
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_out)
        )
        
        # Second conv maintains channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_out)
        )

    def forward(self, x, skip=None):
        # Upsample
        x = self.up(x)
        
        # Handle skip connection
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        return x

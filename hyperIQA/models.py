"""
Neural network architecture definitions for HyperIQA.

This module implements the HyperNetwork-based Image Quality Assessment model,
including the backbone network, hyper network, and target network components.
"""

from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn import init

# Pre-trained model URLs for ResNet variants
MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

# Network architecture constants
RESNET_INITIAL_CHANNELS = 64
RESNET_LAYER_CHANNELS = (64, 128, 256, 512)
RESNET50_LAYER_BLOCKS = (3, 4, 6, 3)
BOTTLENECK_EXPANSION = 4

# HyperNetwork channel reduction constants
HYPER_REDUCTION_CHANNELS = (2048, 1024, 512)

# LDA (Local Distortion Aware) module constants
LDA_POOL_KERNEL_SIZE = 7
LDA1_CONV_CHANNELS = 16
LDA2_CONV_CHANNELS = 32
LDA3_CONV_CHANNELS = 64
LDA1_FC_INPUT_SIZE = LDA1_CONV_CHANNELS * 64  # 16 * 64
LDA2_FC_INPUT_SIZE = LDA2_CONV_CHANNELS * 16  # 32 * 16
LDA3_FC_INPUT_SIZE = LDA3_CONV_CHANNELS * 4   # 64 * 4
LDA4_INPUT_CHANNELS = 2048


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    The hyper network generates weights and biases for the target network
    based on input image features, enabling content-adaptive quality prediction.

    Args:
        lda_out_channels: Local distortion aware module output size.
        hyper_in_channels: Input feature channels for hyper network.
        target_in_size: Input vector size for target network.
        target_fc1_size: First fully connected layer size of target network.
        target_fc2_size: Second fully connected layer size of target network.
        target_fc3_size: Third fully connected layer size of target network.
        target_fc4_size: Fourth fully connected layer size of target network.
        feature_size: Input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy:
        'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.
    """

    def __init__(
        self,
        lda_out_channels: int,
        hyper_in_channels: int,
        target_in_size: int,
        target_fc1_size: int,
        target_fc2_size: int,
        target_fc3_size: int,
        target_fc4_size: int,
        feature_size: int,
    ) -> None:
        super().__init__()

        self.hyper_in_channels = hyper_in_channels
        self.target_in_size = target_in_size
        self.target_fc1_size = target_fc1_size
        self.target_fc2_size = target_fc2_size
        self.target_fc3_size = target_fc3_size
        self.target_fc4_size = target_fc4_size
        self.feature_size = feature_size

        self.backbone = resnet50_backbone(
            lda_out_channels, target_in_size, pretrained=True
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.feature_conv = nn.Sequential(
            nn.Conv2d(HYPER_REDUCTION_CHANNELS[0], HYPER_REDUCTION_CHANNELS[1], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HYPER_REDUCTION_CHANNELS[1], HYPER_REDUCTION_CHANNELS[2], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HYPER_REDUCTION_CHANNELS[2], hyper_in_channels, 1),
            nn.ReLU(inplace=True),
        )

        feature_size_sq = feature_size ** 2

        # Hyper network: conv for generating target fc weights, fc for biases
        self.fc1w_conv = nn.Conv2d(
            hyper_in_channels,
            target_in_size * target_fc1_size // feature_size_sq,
            kernel_size=3,
            padding=1,
        )
        self.fc1b_fc = nn.Linear(hyper_in_channels, target_fc1_size)

        self.fc2w_conv = nn.Conv2d(
            hyper_in_channels,
            target_fc1_size * target_fc2_size // feature_size_sq,
            kernel_size=3,
            padding=1,
        )
        self.fc2b_fc = nn.Linear(hyper_in_channels, target_fc2_size)

        self.fc3w_conv = nn.Conv2d(
            hyper_in_channels,
            target_fc2_size * target_fc3_size // feature_size_sq,
            kernel_size=3,
            padding=1,
        )
        self.fc3b_fc = nn.Linear(hyper_in_channels, target_fc3_size)

        self.fc4w_conv = nn.Conv2d(
            hyper_in_channels,
            target_fc3_size * target_fc4_size // feature_size_sq,
            kernel_size=3,
            padding=1,
        )
        self.fc4b_fc = nn.Linear(hyper_in_channels, target_fc4_size)

        self.fc5w_fc = nn.Linear(hyper_in_channels, target_fc4_size)
        self.fc5b_fc = nn.Linear(hyper_in_channels, 1)

        # Initialize weight generation layers
        self._initialize_hyper_layers()

    def _initialize_hyper_layers(self) -> None:
        """Initialize hyper network layers with Kaiming normal initialization."""
        for name, module in self.named_modules():
            if name.startswith("fc") and hasattr(module, "weight"):
                nn.init.kaiming_normal_(module.weight.data)

    def _get_pooled_features(self, hyper_in_feat: torch.Tensor) -> torch.Tensor:
        """Apply global average pooling and squeeze spatial dimensions."""
        return self.pool(hyper_in_feat).squeeze(-1).squeeze(-1)

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass to generate target network parameters.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            Dictionary containing target network input vector, weights and biases.
        """
        backbone_out = self.backbone(img)

        # Input vector for target net
        target_in_vec = backbone_out["target_in_vec"].view(
            -1, self.target_in_size, 1, 1
        )

        # Input features for hyper net
        hyper_in_feat = self.feature_conv(backbone_out["hyper_in_feat"]).view(
            -1, self.hyper_in_channels, self.feature_size, self.feature_size
        )

        pooled_feat = self._get_pooled_features(hyper_in_feat)

        # Generate target net weights & biases
        return {
            "target_in_vec": target_in_vec,
            "target_fc1w": self.fc1w_conv(hyper_in_feat).view(
                -1, self.target_fc1_size, self.target_in_size, 1, 1
            ),
            "target_fc1b": self.fc1b_fc(pooled_feat).view(-1, self.target_fc1_size),
            "target_fc2w": self.fc2w_conv(hyper_in_feat).view(
                -1, self.target_fc2_size, self.target_fc1_size, 1, 1
            ),
            "target_fc2b": self.fc2b_fc(pooled_feat).view(-1, self.target_fc2_size),
            "target_fc3w": self.fc3w_conv(hyper_in_feat).view(
                -1, self.target_fc3_size, self.target_fc2_size, 1, 1
            ),
            "target_fc3b": self.fc3b_fc(pooled_feat).view(-1, self.target_fc3_size),
            "target_fc4w": self.fc4w_conv(hyper_in_feat).view(
                -1, self.target_fc4_size, self.target_fc3_size, 1, 1
            ),
            "target_fc4b": self.fc4b_fc(pooled_feat).view(-1, self.target_fc4_size),
            "target_fc5w": self.fc5w_fc(pooled_feat).view(
                -1, 1, self.target_fc4_size, 1, 1
            ),
            "target_fc5b": self.fc5b_fc(pooled_feat).view(-1, 1),
        }


class TargetNet(nn.Module):
    """
    Target network for quality prediction.

    Uses dynamically generated weights from HyperNet to predict image quality.
    """

    def __init__(self, params: dict[str, torch.Tensor]) -> None:
        """
        Initialize target network with hyper-generated parameters.

        Args:
            params: Dictionary containing weights and biases from HyperNet.
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            TargetFC(params["target_fc1w"], params["target_fc1b"]),
            nn.Sigmoid(),
        )
        self.layer2 = nn.Sequential(
            TargetFC(params["target_fc2w"], params["target_fc2b"]),
            nn.Sigmoid(),
        )
        self.layer3 = nn.Sequential(
            TargetFC(params["target_fc3w"], params["target_fc3b"]),
            nn.Sigmoid(),
        )
        self.layer4 = nn.Sequential(
            TargetFC(params["target_fc4w"], params["target_fc4b"]),
            nn.Sigmoid(),
            TargetFC(params["target_fc5w"], params["target_fc5b"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quality prediction.

        Args:
            x: Input feature tensor from HyperNet.

        Returns:
            Quality score tensor.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out).squeeze()
        return out


class TargetFC(nn.Module):
    """
    Fully connected layer with per-sample weights for target network.

    Uses group convolution to apply different weights/biases for each
    image in a batch, enabling content-adaptive processing.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initialize with per-sample weights and biases.

        Args:
            weight: Weight tensor of shape (B, out_features, in_features, 1, 1).
            bias: Bias tensor of shape (B, out_features).
        """
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply per-sample fully connected operation using group convolution.

        Args:
            input_tensor: Input tensor of shape (B, in_features, 1, 1).

        Returns:
            Output tensor of shape (B, out_features, 1, 1).
        """
        batch_size = input_tensor.shape[0]
        in_channels = input_tensor.shape[1]
        height, width = input_tensor.shape[2], input_tensor.shape[3]

        # Reshape for group convolution
        input_reshaped = input_tensor.view(1, batch_size * in_channels, height, width)
        weight_reshaped = self.weight.view(
            batch_size * self.weight.shape[1],
            self.weight.shape[2],
            self.weight.shape[3],
            self.weight.shape[4],
        )
        bias_reshaped = self.bias.view(-1)

        out = F.conv2d(
            input=input_reshaped,
            weight=weight_reshaped,
            bias=bias_reshaped,
            groups=batch_size,
        )

        return out.view(batch_size, self.weight.shape[1], height, width)


class Bottleneck(nn.Module):
    """
    ResNet bottleneck block with expansion factor.

    Implements the standard bottleneck architecture with 1x1 -> 3x3 -> 1x1
    convolutions and skip connection.
    """

    expansion = BOTTLENECK_EXPANSION

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """
        Initialize bottleneck block.

        Args:
            inplanes: Number of input channels.
            planes: Number of intermediate channels.
            stride: Stride for the 3x3 convolution.
            downsample: Optional downsampling layer for skip connection.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with skip connection added.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """
    ResNet backbone with Local Distortion Aware (LDA) modules.

    Extracts multi-scale features from the input image and produces
    both hyper network input features and target network input vector.
    """

    def __init__(
        self,
        lda_out_channels: int,
        target_in_size: int,
        block: type[Bottleneck],
        layers: tuple[int, ...],
        num_classes: int = 1000,
    ) -> None:
        """
        Initialize ResNet backbone with LDA modules.

        Args:
            lda_out_channels: Output channels for each LDA module.
            target_in_size: Total input size for target network.
            block: Block type (Bottleneck).
            layers: Number of blocks in each layer.
            num_classes: Number of output classes (unused, for compatibility).
        """
        super().__init__()
        self.inplanes = RESNET_INITIAL_CHANNELS

        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(
            3, RESNET_INITIAL_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(RESNET_INITIAL_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, RESNET_LAYER_CHANNELS[0], layers[0])
        self.layer2 = self._make_layer(
            block, RESNET_LAYER_CHANNELS[1], layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, RESNET_LAYER_CHANNELS[2], layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, RESNET_LAYER_CHANNELS[3], layers[3], stride=2
        )

        # Local Distortion Aware modules
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, LDA1_CONV_CHANNELS, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(LDA_POOL_KERNEL_SIZE, stride=LDA_POOL_KERNEL_SIZE),
        )
        self.lda1_fc = nn.Linear(LDA1_FC_INPUT_SIZE, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, LDA2_CONV_CHANNELS, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(LDA_POOL_KERNEL_SIZE, stride=LDA_POOL_KERNEL_SIZE),
        )
        self.lda2_fc = nn.Linear(LDA2_FC_INPUT_SIZE, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, LDA3_CONV_CHANNELS, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(LDA_POOL_KERNEL_SIZE, stride=LDA_POOL_KERNEL_SIZE),
        )
        self.lda3_fc = nn.Linear(LDA3_FC_INPUT_SIZE, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(LDA_POOL_KERNEL_SIZE, stride=LDA_POOL_KERNEL_SIZE)
        self.lda4_fc = nn.Linear(LDA4_INPUT_CHANNELS, target_in_size - lda_out_channels * 3)

        self._initialize_weights()
        self._initialize_lda_layers()

    def _initialize_weights(self) -> None:
        """Initialize convolutional and batch norm layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _initialize_lda_layers(self) -> None:
        """Initialize LDA module layers with Kaiming normal initialization."""
        nn.init.kaiming_normal_(self.lda1_pool[0].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool[0].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool[0].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Create a ResNet layer with specified number of blocks.

        Args:
            block: Block type to use.
            planes: Number of output channels.
            blocks: Number of blocks in the layer.
            stride: Stride for the first block.

        Returns:
            Sequential container of blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through backbone network.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing 'hyper_in_feat' and 'target_in_vec'.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Apply LDA modules at different scales
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        target_in_vec = torch.cat((lda_1, lda_2, lda_3, lda_4), dim=1)

        return {
            "hyper_in_feat": x,
            "target_in_vec": target_in_vec,
        }


def resnet50_backbone(
    lda_out_channels: int,
    target_in_size: int,
    pretrained: bool = False,
    **kwargs: Any,
) -> ResNetBackbone:
    """
    Construct a ResNet-50 backbone model.

    Args:
        lda_out_channels: Output channels for LDA modules.
        target_in_size: Input size for target network.
        pretrained: If True, loads weights pre-trained on ImageNet.
        **kwargs: Additional arguments passed to ResNetBackbone.

    Returns:
        ResNetBackbone model instance.
    """
    model = ResNetBackbone(
        lda_out_channels, target_in_size, Bottleneck, RESNET50_LAYER_BLOCKS, **kwargs
    )
    if pretrained:
        pretrained_dict = model_zoo.load_url(MODEL_URLS["resnet50"])
        model_dict = model.state_dict()
        # Filter out incompatible keys
        state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(module: nn.Module) -> None:
    """
    Initialize module weights using Xavier/Kaiming initialization.

    Args:
        module: Neural network module to initialize.
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        init.kaiming_normal_(module.weight.data)
    elif "Linear" in classname:
        init.kaiming_normal_(module.weight.data)
    elif "BatchNorm2d" in classname:
        init.uniform_(module.weight.data, 0.02, 1.0)
        init.constant_(module.bias.data, 0.0)

"""
MS-HyperIQA: Multi-Scale HyperIQA with Feature Pyramid and Attention

Enhanced version of HyperIQA with:
1. Multi-scale feature extraction (ResNet layers 2, 3, 4)
2. Feature Pyramid Network (FPN) for feature fusion
3. Channel and Spatial Attention mechanisms
4. Improved training strategies
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-Net style)

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.

    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    Focuses on 'where' is informative by generating a spatial attention map
    that emphasizes or suppresses features in different spatial locations.

    Args:
        kernel_size: Size of the convolutional kernel
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    Sequentially applies channel and spatial attention mechanisms.

    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion

    Builds a feature pyramid with top-down pathway and lateral connections
    to create semantically strong features at all scales.

    Args:
        in_channels_list: List of input channel numbers [C2, C3, C4]
        out_channels: Output channel number for all pyramid levels
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_conv2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_conv3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_conv4 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Smoothing layers (3x3 conv to reduce aliasing)
        self.smooth_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _upsample_add(self, x, y):
        """Upsample x and add to y"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, c2, c3, c4):
        """
        Args:
            c2: Layer 2 features (low-level, high resolution)
            c3: Layer 3 features (mid-level)
            c4: Layer 4 features (high-level, low resolution)

        Returns:
            p2, p3, p4: Pyramid features at different scales
        """
        # Build top-down pathway
        p4 = self.lateral_conv4(c4)
        p3 = self._upsample_add(p4, self.lateral_conv3(c3))
        p2 = self._upsample_add(p3, self.lateral_conv2(c2))

        # Apply smoothing
        p4 = self.smooth_conv4(p4)
        p3 = self.smooth_conv3(p3)
        p2 = self.smooth_conv2(p2)

        return p2, p3, p4


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class MultiScaleResNetBackbone(nn.Module):
    """
    Multi-Scale ResNet Backbone with LDA modules

    Extracts features at multiple scales (layers 2, 3, 4) for comprehensive
    representation of image quality at different receptive field sizes.
    """
    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(MultiScaleResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Local distortion aware modules for different layers
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Initialize LDA modules
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Extract multi-scale features
        # Layer 2 features: 256 channels, 1/8 resolution
        c2 = self.layer2(x)
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))

        # Layer 3 features: 512 channels, 1/16 resolution
        c3 = self.layer3(c2)
        lda_2 = self.lda2_fc(self.lda2_pool(c2).view(c2.size(0), -1))

        # Layer 4 features: 1024 channels, 1/32 resolution
        c4 = self.layer4(c3)
        lda_3 = self.lda3_fc(self.lda3_pool(c3).view(c3.size(0), -1))
        lda_4 = self.lda4_fc(self.lda4_pool(c4).view(c4.size(0), -1))

        # Concatenate LDA features
        lda_vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        out = {}
        out['c2'] = c2  # 512-dim
        out['c3'] = c3  # 1024-dim
        out['c4'] = c4  # 2048-dim
        out['lda_vec'] = lda_vec  # 224-dim

        return out


class MSHyperNet(nn.Module):
    """
    Multi-Scale HyperNet with Feature Pyramid and Attention

    Enhanced HyperNet that leverages multi-scale features through FPN
    and attention mechanisms for generating image-specific TargetNet weights.

    Args:
        lda_out_channels: Output size for each LDA module
        hyper_in_channels: Input feature channels for hyper network
        target_in_size: Input vector size for target network (LDA vector)
        target_fc*_size: Fully connected layer sizes for target network
        feature_size: Feature map spatial size for hyper network
        fpn_channels: Number of channels for FPN outputs
    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size,
                 target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size,
                 feature_size, fpn_channels=256):
        super(MSHyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size
        self.fpn_channels = fpn_channels

        # Multi-scale ResNet backbone
        self.res = multiscale_resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], fpn_channels)

        # Attention modules for each pyramid level
        self.attention_p2 = CBAM(fpn_channels)
        self.attention_p3 = CBAM(fpn_channels)
        self.attention_p4 = CBAM(fpn_channels)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature fusion: merge multi-scale features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * 3, fpn_channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, self.hyperInChn, 1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Hyper network part: conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3, padding=1)
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=1)
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=1)
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=1)
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # Initialize
        for m in self.fusion_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

        for i, m_name in enumerate(self._modules):
            if 'fc' in m_name and hasattr(self._modules[m_name], 'weight'):
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        # Extract multi-scale features
        res_out = self.res(img)
        c2, c3, c4 = res_out['c2'], res_out['c3'], res_out['c4']

        # Build feature pyramid
        p2, p3, p4 = self.fpn(c2, c3, c4)

        # Apply attention to each pyramid level
        p2 = self.attention_p2(p2)
        p3 = self.attention_p3(p3)
        p4 = self.attention_p4(p4)

        # Resize all to same spatial size (use p4's size)
        _, _, h, w = p4.size()
        p2_resized = F.adaptive_avg_pool2d(p2, (h, w))
        p3_resized = F.adaptive_avg_pool2d(p3, (h, w))

        # Concatenate multi-scale features
        multi_scale_feat = torch.cat([p2_resized, p3_resized, p4], dim=1)

        # Fuse features
        hyper_in_feat = self.fusion_conv(multi_scale_feat)
        hyper_in_feat = F.adaptive_avg_pool2d(hyper_in_feat, (feature_size, feature_size))

        # Input vector for target net (LDA vector)
        target_in_vec = res_out['lda_vec'].view(-1, self.target_in_size, 1, 1)

        # Generate target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out


class TargetNet(nn.Module):
    """
    Target network for quality prediction.

    Note: This is kept the same as original HyperIQA for compatibility.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        # Enable dropout for regularization
        self.dropout = nn.Dropout(0.5)

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        q = self.dropout(q)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


def multiscale_resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """
    Constructs a Multi-Scale ResNet-50 backbone.

    Args:
        lda_out_channels: Output channels for each LDA module
        in_chn: Total LDA vector size
        pretrained: If True, returns a model pre-trained on ImageNet
    """
    model = MultiScaleResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    """Test the model architecture"""
    print("Testing MS-HyperIQA Model Architecture...")
    print("=" * 60)

    # Create model
    model = MSHyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()

    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224).cuda()

    print(f"Input shape: {test_input.shape}")
    print()

    # Forward pass through HyperNet
    print("Forward pass through MSHyperNet...")
    paras = model(test_input)

    print(f"  Generated parameters for TargetNet:")
    print(f"    - target_in_vec: {paras['target_in_vec'].shape}")
    print(f"    - target_fc1w: {paras['target_fc1w'].shape}")
    print(f"    - target_fc1b: {paras['target_fc1b'].shape}")
    print()

    # Forward pass through TargetNet
    print("Forward pass through TargetNet...")
    model_target = TargetNet(paras).cuda()
    pred = model_target(paras['target_in_vec'])

    print(f"  Quality predictions shape: {pred.shape}")
    print(f"  Quality scores: {pred}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    print("✓ Model architecture test passed!")

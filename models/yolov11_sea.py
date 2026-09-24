"""
YOLOv11-SEA: Semantic Segmentation Architecture for Unfouled Aperture Extraction
Integrated with Parameter-Free Simple Attention Module (SimAM)
and Laplacian of Gaussian (LoG) Edge Boundary Loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Parameter-Free Simple Attention Module (SimAM)
# ============================================================================

class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks.
    Computes 3D attention weights based on energy function optimization per neuron.
    """
    def __init__(self, e_lambda: float = 1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Spatial variance from channel mean
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # Energy function formula: y = (x - mu)^2 / (4 * (sigma^2 + lambda)) + 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activaton(y)


# ============================================================================
# 2. Laplacian of Gaussian (LoG) Edge Loss
# ============================================================================

class LoGEdgeLoss(nn.Module):
    """
    Laplacian of Gaussian (LoG) boundary supervision loss.
    Extracts high-frequency twine boundaries to sharpen spatial delineation.
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, tau: float = 0.008):
        super(LoGEdgeLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.tau = tau
        
        # Build 2D LoG kernel as formulated in Eq. (1)
        k_radius = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-k_radius, k_radius + 1, dtype=torch.float32),
            torch.arange(-k_radius, k_radius + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        # LoG(x, y, sigma) = ((x^2 + y^2 - 2*sigma^2) / sigma^4) * (1 / (2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
        norm_factor = 1.0 / (2.0 * math.pi * (sigma ** 2))
        exponent = torch.exp(-(x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))
        laplace_factor = (x ** 2 + y ** 2 - 2.0 * (sigma ** 2)) / (sigma ** 4)
        log_kernel = laplace_factor * norm_factor * exponent
        
        # Normalize kernel sum to 0
        log_kernel = log_kernel - log_kernel.mean()
        self.register_buffer('log_kernel', log_kernel.view(1, 1, kernel_size, kernel_size))

    def extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        edge_resp = F.conv2d(x_pad, self.log_kernel)
        edge_mask = (torch.abs(edge_resp) > self.tau).float()
        return edge_mask

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes edge-weighted BCE loss on high-frequency boundary response zones.
        """
        pred_prob = torch.sigmoid(pred_logits)
        target_edges = self.extract_edges(target_mask.float())
        
        # Weighted boundary loss
        bce_loss = F.binary_cross_entropy(pred_prob, target_mask.float(), reduction='none')
        edge_weighted_loss = (bce_loss * (1.0 + 2.0 * target_edges)).mean()
        return edge_weighted_loss


# ============================================================================
# 3. Dedicated Semantic Segmentation Head without Bounding Box Decoders
# ============================================================================

class ConvBlock(nn.Module):
    """Standard Convolution-BatchNorm-SiLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOv11_SEA_Head(nn.Module):
    """
    Dedicated semantic segmentation head eliminating all bounding box detection branches,
    object anchors, and redundant decoders.
    Directly predicts single-channel binary logits for unfouled aperture extraction.
    """
    def __init__(self, in_channels_list: list = [64, 128, 256], mid_channels: int = 64):
        super(YOLOv11_SEA_Head, self).__init__()
        p3_ch, p4_ch, p5_ch = in_channels_list
        
        # Multi-scale feature refinement with SimAM attention
        self.p5_conv = ConvBlock(p5_ch, mid_channels, 1, 1, 0)
        self.p4_conv = ConvBlock(p4_ch, mid_channels, 1, 1, 0)
        self.p3_conv = ConvBlock(p3_ch, mid_channels, 1, 1, 0)
        
        self.simam5 = SimAM()
        self.simam4 = SimAM()
        self.simam3 = SimAM()
        
        # Feature aggregation decoder
        self.fusion_conv = nn.Sequential(
            ConvBlock(mid_channels * 3, mid_channels, 3, 1, 1),
            ConvBlock(mid_channels, mid_channels // 2, 3, 1, 1)
        )
        
        # Final semantic mask logit generator (Single class: Aperture foreground)
        self.out_conv = nn.Conv2d(mid_channels // 2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor, target_size: tuple) -> torch.Tensor:
        h, w = target_size
        
        # Project and apply parameter-free SimAM attention
        f5 = self.simam5(self.p5_conv(p5))
        f4 = self.simam4(self.p4_conv(p4))
        f3 = self.simam3(self.p3_conv(p3))
        
        # Multi-scale spatial alignment via bilinear upsampling
        f5_up = F.interpolate(f5, size=(h, w), mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=(h, w), mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale representations
        feat_cat = torch.cat([f3_up, f4_up, f5_up], dim=1)
        fused = self.fusion_conv(feat_cat)
        logits = self.out_conv(fused)
        
        return logits


class YOLOv11_SEA(nn.Module):
    """
    Complete YOLOv11-SEA lightweight semantic segmentation model.
    Takes underwater RGB images and outputs clean aperture binary masks.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3):
        super(YOLOv11_SEA, self).__init__()
        
        # Efficient lightweight convolutional backbone
        self.stem = ConvBlock(in_channels, 32, 3, 2, 1) # 1/2
        self.stage1 = ConvBlock(32, 64, 3, 2, 1)        # 1/4 (P3)
        self.stage2 = ConvBlock(64, 128, 3, 2, 1)       # 1/8 (P4)
        self.stage3 = ConvBlock(128, 256, 3, 2, 1)      # 1/16 (P5)
        
        # Attention-enhanced semantic head
        self.seg_head = YOLOv11_SEA_Head(in_channels_list=[64, 128, 256], mid_channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.shape[2], x.shape[3])
        
        # Forward backbone
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        
        # Forward dedicated semantic head
        logits = self.seg_head(p3, p4, p5, target_size=input_size)
        return logits

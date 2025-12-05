import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class SE(nn.Module):
    def __init__(self, c1, r=16):  # c1: 输入通道数，r: 压缩比例
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(c1 // r, c1, bias=False),   # 升维
            nn.Sigmoid()                          # 归一化
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 保留尺寸兼容性

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))  # 使用普通卷积
        out = self.pool(out)
        shortcut = self.shortcut(x)
        # 调整 shortcut 尺寸以匹配 out（需使用 interpolate 或 padding）
        if shortcut.shape[2:] != out.shape[2:]:
            shortcut = F.interpolate(shortcut, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + shortcut

class Deformable_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * 3 * 3, kernel_size=3, padding=1
        )
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels) 
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 保留尺寸兼容性

    def forward(self, x):
        offset = self.offset_conv(x)
        out = F.relu(self.bn(self.deform_conv(x,offset)))
        out = self.se(out)
        out = self.pool(out)
        shortcut = self.shortcut(x)
        # 调整 shortcut 尺寸以匹配 out（需使用 interpolate 或 padding）
        if shortcut.shape[2:] != out.shape[2:]:
            shortcut = F.interpolate(shortcut, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + shortcut


class ImageFeatExactModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_ratio=0.):
        super(ImageFeatExactModule, self).__init__()
        
        self.Exact_Backbone=nn.Sequential(
            ResBlock(in_channels, 64),
            Deformable_ResBlock(64, 128),
            Deformable_ResBlock(128, 256),
            Deformable_ResBlock(256, 512),
            nn.Dropout2d(dropout_ratio)
        )
        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.Exact_Backbone(x)
        x = self.final_conv(x)
        return F.relu(x)


class FeatCrossModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatCrossModule, self).__init__()
        self.theta = nn.Parameter(torch.ones(1, 1, in_channels))
        
        # 交叉注意力模块
        self.cross_attn1 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=False)
        self.cross_attn2 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=False)
        
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.up_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(), 
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.GELU() 
        )

    def forward(self, vit_early_feature, image_exact_feature):
        b, c, h, w = image_exact_feature.shape
        # 调整维度以适应多头注意力机制
        vit_early_feature = vit_early_feature.permute(1, 0, 2)  # [N, B, C]
        image_exact_feature_flat = image_exact_feature.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # 第一次交叉注意力：ViT特征作为查询，CNN特征作为键值对
        attn_output1, _ = self.cross_attn1(
            query=vit_early_feature,
            key=image_exact_feature_flat,
            value=image_exact_feature_flat
        )
        
        # 残差连接和层归一化
        updated_vit_feature = vit_early_feature + self.theta * attn_output1
        updated_vit_feature = self.norm1(updated_vit_feature)
        
        # 第二次交叉注意力
        seq_len = updated_vit_feature.size(0)
        if image_exact_feature_flat.size(0) != seq_len:
            image_exact_feature_proj = nn.functional.adaptive_avg_pool1d(
                image_exact_feature_flat.permute(1, 2, 0), seq_len
            ).permute(2, 0, 1)
        else:
            image_exact_feature_proj = image_exact_feature_flat
        
        attn_output2, _ = self.cross_attn2(
            query=image_exact_feature_proj,
            key=updated_vit_feature,
            value=updated_vit_feature
        )
        
        # 残差连接和层归一化
        updated_spatial_feature = image_exact_feature_proj + attn_output2
        updated_spatial_feature = self.norm2(updated_spatial_feature)
        
        # 重塑回空间维度
        updated_spatial_feature = updated_spatial_feature.permute(1, 2, 0).view(b, c, h, w)
        
        # 上采样
        HQ_feature = self.up_path(updated_spatial_feature)
        
        return HQ_feature



# class FeatCrossModule(nn.Module):
#     def __init__(self, in_channels):
#         super(FeatCrossModule, self).__init__()
#         self.theta = nn.Parameter(torch.ones(1))
#         self.cross_attn1 = nn.MultiheadAttention(in_channels, num_heads=8)
#         self.cross_attn2 = nn.MultiheadAttention(in_channels, num_heads=8)
#         self.up_conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.bn_up1 = nn.BatchNorm2d(in_channels // 2)
#         self.up_conv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
#         self.bn_up2 = nn.BatchNorm2d(in_channels // 4)

#     def forward(self, vit_early_feature, image_exact_feature):
#         b, c, h, w = image_exact_feature.shape
#         # 调整维度以适应多头注意力机制
#         vit_early_feature = vit_early_feature.permute(1, 0, 2)
#         image_exact_feature = image_exact_feature.flatten(2).permute(2, 0, 1)

#         # 第一次交叉注意力
#         attn_output1, _ = self.cross_attn1(vit_early_feature, image_exact_feature, image_exact_feature)
#         updated_vit_feature = vit_early_feature + self.theta * attn_output1

#         # 第二次交叉注意力
#         attn_output2, _ = self.cross_attn2(image_exact_feature, updated_vit_feature, updated_vit_feature)
#         updated_spatial_feature = image_exact_feature + attn_output2

#         # 上采样
#         updated_spatial_feature = updated_spatial_feature.permute(1, 2, 0).view(b, c, h, w)
#         upsampled_feature = F.relu(self.bn_up1(self.up_conv1(updated_spatial_feature)))
#         upsampled_feature = F.relu(self.bn_up2(self.up_conv2(upsampled_feature)))

#         return upsampled_feature


# class ClickAdapter(nn.Module):
#     def __init__(self, in_channels=3, out_channels=256):
#         super(ClickAdapter, self).__init__()
#         self.spatial_prior_module = ImageFeatExactModule(in_channels, out_channels)
#         self.injector_module = FeatCrossModule(out_channels)

#     def forward(self, vit_early_feature, input_image):
#         spatial_prior_feature = self.spatial_prior_module(input_image)
#         high_quality_feature = self.injector_module(vit_early_feature, spatial_prior_feature)
#         return high_quality_feature


    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from isegm.model.modeling.transformer_helper.wrappers import resize
from mmcv.cnn import ConvModule
import numpy as np
from isegm.model.modeling.detail_feature_exact import ImageFeatExactModule, FeatCrossModule
# from isegm.model.frozen_is_model_prevMod import XConvBnRelu2


class XConvBnRelu2(nn.Module):
    """
    Xception conv bn relu
    """

    def __init__(self, input_dims=3, out_dims=16, **kwargs):
        super(XConvBnRelu2, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input_dims, input_dims, 3, 1, 1, groups=input_dims)
        self.norm1 = nn.BatchNorm2d(input_dims)
        self.conv3x3_2 = nn.Conv2d(input_dims, input_dims, 3, 1, 1, groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims, out_dims, 1, 1, 0)
        self.norm2 = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3_1(x)
        x = self.norm1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(DecoderLayer, self).__init__()
        self.image_to_token_attn = nn.MultiheadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.),
        )
        self.token_to_image_attn = nn.MultiheadAttention(d_model, n_heads)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, mask_token, feature):
        # Token to Image Attention
        attn_output, _ = self.token_to_image_attn(mask_token, feature, feature)
        mask_token = self.norm1(mask_token + attn_output)
        
        # Self Attention
        attn_output, _ = self.self_attn(mask_token, mask_token, mask_token)
        mask_token = self.norm2(mask_token + attn_output)

        # MLP
        mlp_output = self.mlp(mask_token)
        mask_token = self.norm3(mask_token + mlp_output)
        
        # Image to Token Attention
        attn_output, _ = self.image_to_token_attn(feature, mask_token, mask_token)
        feature = self.norm4(feature + attn_output)

        return feature, mask_token

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(MultiScaleFusion, self).__init__()
        self.in_channels=in_channels
        self.out_channel=out_channel
        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channel,
                    kernel_size=1,
                    stride=1,
                )
            )
        self.fusion_conv = ConvModule(
            in_channels=self.out_channel * len(self.in_channels),
            out_channels=self.out_channel,
            kernel_size=1,
        )
        
        
    def forward(self, crossed_feat, multi_scale_features):
        outs = []
        for idx in range(len(multi_scale_features)):
            x = multi_scale_features[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=crossed_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False))
        cat_multi_feature = self.fusion_conv(torch.cat(outs, dim=1))
        return crossed_feat + cat_multi_feature
        # return cat_multi_feature


class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposedConvBlock, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(in_channels, out_channels*2, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x=self.t_conv1(x)
        x=self.t_conv2(x)
        
        return x


class PreviousMaskProcessing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreviousMaskProcessing, self).__init__()
        self.downscale = nn.Sequential(
            ConvModule(
                in_channels=in_channels,     
                out_channels=64,     
                kernel_size=3, 
                stride=2, 
                padding=1,
            ),
            ConvModule(
                in_channels=64,    
                out_channels=128,    
                kernel_size=3, 
                stride=2, 
                padding=1,
            ),
            ConvModule(
                in_channels=128,    
                out_channels=256,    
                kernel_size=3, 
                stride=2, 
                padding=1,
            ),
            ConvModule(
                in_channels=256,    
                out_channels=512,    
                kernel_size=3, 
                stride=2, 
                padding=1,
            ),
            XConvBnRelu2(512, out_channels),
        )

    def forward(self, coord_features):
        return self.downscale(coord_features)

class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, token_dim):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.kernel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim,
                      out_channels * in_channels * kernel_size * kernel_size
                      + out_channels)
        )

    def forward(self, feat, token):
        """
        feat: [B, in_c, H, W]
        token: [B, token_dim]
        """
        B, C, H, W = feat.shape
        params = self.kernel_mlp(token)  # [B, out_c*in_c*k*k + out_c]
        weight_len = self.out_channels * self.in_channels * self.k * self.k
        weight = params[:, :weight_len]
        bias   = params[:, weight_len:]
        weight = weight.view(B * self.out_channels,
                             self.in_channels,
                             self.k, self.k)
        bias   = bias.view(B * self.out_channels)
        feat_group = feat.view(1, B*self.in_channels, H, W)
        out = F.conv2d(
            feat_group,
            weight,
            bias=bias,
            stride=1,
            padding=self.k//2,
            groups=B
        )
        out = out.view(B, self.out_channels, H, W)
        return out

class FinalTokenDecoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super(FinalTokenDecoder, self).__init__()
        self.token_to_image_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(d_model // 2, d_model // 4),
            nn.Dropout(0.),
        )
    def forward(self, token, image_feature):
        attn_output, _ = self.token_to_image_attn(token, image_feature, image_feature)
        token = self.norm(token + attn_output)
        return self.mlp(token)


class Refiner(nn.Module):
    def __init__(self, in_channels=[128, 256, 512, 1024], in_index=[0, 1, 2, 3], dropout_ratio=0.1,
        num_classes=1, align_corners=False, d_model=768, n_heads=8):
        super(Refiner, self).__init__()
        self.in_channels=in_channels
        self.exact_image_feature=ImageFeatExactModule(3, d_model)
        
        self.dense_fusion_conv = nn.Conv2d(2 * d_model, d_model, kernel_size=1)
        self.cross_feat = FeatCrossModule(d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(2)])
        self.final_token_decoder=FinalTokenDecoder(d_model, n_heads)
        self.multi_scale_fusion = MultiScaleFusion(self.in_channels, d_model//4)
        self.transposed_conv = TransposedConvBlock(d_model, d_model//4)
        self.prev_mask_downscaling = PreviousMaskProcessing(4, d_model)  # image(3)、prev_mask(1)、prev_mask_modualte(1)
        self.dynamic_conv = DynamicConv(
            in_channels = d_model//4,
            out_channels = num_classes,      
            kernel_size = 3,                 
            token_dim   = d_model // 4          
        )
        
    def forward(self, vit_early_feature, multi_scale_features, backbone_vit_feature, coord_features, image, prev_mask, prev_mask_modulated, mask_token):
        image_exact_feature = self.exact_image_feature(image)  # convnet exacts spatial image feature
        image_crossed_feat = self.cross_feat(vit_early_feature, image_exact_feature) # crossattention merges vit early feature and spatial image feature
        prev_mask_downscaled = self.prev_mask_downscaling(coord_features)
        mask_feature=torch.cat([prev_mask_downscaled, backbone_vit_feature], dim=1)
        mask_feature = self.dense_fusion_conv(mask_feature)
        mask_feature = prev_mask_downscaled.flatten(2).permute(2, 0, 1)  
        for decoder_layer in self.decoder_layers:
            mask_feature, mask_token = decoder_layer(mask_token, mask_feature)
            
        mask_token = self.final_token_decoder(mask_token, mask_feature)
        mask_token = mask_token.view(mask_token.size(1), -1)
        
        N, B, C=mask_feature.shape[0],mask_feature.shape[1],mask_feature.shape[2]
        mask_feature=mask_feature.view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # multi-scale fusion
        fused_feature = self.multi_scale_fusion(image_crossed_feat, multi_scale_features)
        # transposed convolution
        transposed_mask_feature = self.transposed_conv(mask_feature)
        final_feature = transposed_mask_feature + fused_feature
        mask = self.dynamic_conv(final_feature, mask_token)
    
        return {'instances': mask, 'instances_aux': None}




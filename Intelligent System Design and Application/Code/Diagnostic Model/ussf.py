import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.fft
from einops import rearrange, repeat

from visualizer import get_local
get_local.activate()

MIN_NUM_PATCHES = 16

table = [
            # t, c, n, s, SE
            [1,  24,  2, 1, 0],
            [4,  48,  4, 2, 0],
            [4,  64,  4, 2, 0],
            [4, 128,  6, 2, 1],
            #[6, 160,  9, 1, 1],
            #[6, 256, 15, 2, 1],
        ]


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        SiLU()
    )

class FDCF(nn.Module):
    def __init__(self, hidden_size):
        super(FDCF, self).__init__()

        # Convolution layer as per the diagram
        self.conv = nn.Conv2d(3, hidden_size, kernel_size=3, padding=1)

        # Gaussian High-Pass Filter (simplified for demonstration)
        self.gh_pf = self.gaussian_high_pass_filter()

        # Max Pooling layer
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def gaussian_high_pass_filter(self):
        # Creating a simple Gaussian high-pass filter
        # For simplicity, we'll define a 3x3 kernel (this can be more complex)
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)

    def forward(self, x):
        # Apply FFT to the input
        fft_x = torch.fft.fftn(x)

        # Take the absolute value (magnitude) of the complex tensor
        abs_fft_x = torch.abs(fft_x)  # This converts complex to real (magnitude)

        # Apply convolution to the magnitude of the FFT
        x = self.conv(abs_fft_x)

        # Apply Gaussian High-Pass Filter
        gh_x = self.gh_pf(x)

        # Inverse FFT to get back to the spatial domain
        ifft_x = torch.fft.ifftn(gh_x)

        # Take the absolute value of the inverse FFT result
        abs_x = torch.abs(ifft_x)

        # Max pooling
        x = self.mp(abs_x)

        return x




class SimpleConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 2, 2) if downsample else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return F.relu(self.block(x) + x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_qkv_h = nn.Linear(dim, dim * 3)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
                            Rearrange('b h i j -> b i j h'),
                            nn.LayerNorm(heads),
                            Rearrange('b i j h -> b h i j'))

        self.elu = nn.ELU(alpha=1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),)

    def forward(self, x, hidden):
        b, n, _, h = *x.shape, self.heads

        qkv = torch.add(self.to_qkv(x), self.to_qkv_h(hidden)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #out = self.linear_attn(self._elu(q), self._elu(k), v)
        #out = self.re_attn(q, k, v)
        out = self.softmax_attn(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

    def _elu(self, x):
        return self.elu(x) + 1

    #@get_local('attn')
    def softmax_attn(self, q, k, v):
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out
    @get_local('attn')
    def linear_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out

    def re_attn(self, q, k, v):
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = torch.einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1, deterministic=True, rnn_layer = True):
        super().__init__()

        self.rnn_layer = rnn_layer
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_hidden = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        self.attention = MultiHeadDotProductAttention(input_shape, heads = heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention  = nn.Dropout(attention_dropout_rate)

    def forward(self, x, h):

        residual_x, residual_h = x, h
        attn = self.attention(self.layer_norm_input(x), self.layer_norm_hidden(h))
        x = self.drop_out_attention(attn) + residual_x

        residual_x = x
        x = self.layer_norm_out(x)
        x = self.mlp(x)
        x += residual_x

        return x, attn + residual_h


class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        # encoder blocks
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

    def forward(self, x, hidden, pos_embedding):
        h = hidden.clone()
        for i, layer in enumerate(self.layers):
            x, h[i] = layer[0](self.dropout(x         + pos_embedding),
                                           (hidden[i] + pos_embedding),)

        return x, h

class US_RViT(nn.Module):
    """ Vision Transformer with SlowFast and Frequency Fusion for Video Classification """
    def __init__(self, *, image_size, patch_size, num_classes, depth, length, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0., batch):
        super(US_RViT, self).__init__()
        self.length = length
        self.batch = batch
        num_patches = (image_size // patch_size) ** 2
        hidden_size = channels * patch_size ** 2

        self.slow_length = int (length / 2)
        self.fast_length = int (length / 5)
        self.embedding = nn.Conv2d(channels, hidden_size, patch_size, patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self._cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self._seq = nn.Parameter(torch.randn(depth, 1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)
        # self.cross = CrossAttentionFusion(hidden_size=hidden_size,num_heads=8)
        # Transformers
        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate = dropout)

        # FDCF
        self.fdcf = FDCF(channels)
        self.deconv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)

        # Multi-Head Attention for Fusion
        self.fusion_mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=heads, batch_first=True)
        self.conv_fuse = nn.Conv2d(in_channels=self.batch, out_channels=self.batch, kernel_size=1)
        self.to_cls_token = nn.Identity()
        # Classifier
        self.fc = FC_class(hidden_size * (depth + 1), num_classes)

    def attention_weighted_fusion(self, a, b, c):
        B, N, C = a.shape
        X = c.shape[1]

        # 将 c 映射为 (B, 1, C)，作为权重
        c_weight = c.mean(dim=1, keepdim=True)  # (B, 1, C)

        # 生成 attention gate
        alpha = torch.sigmoid(c_weight)  # (B, 1, C)

        fused = alpha * a + (1 - alpha) * b
        return fused  # (B, 785, C)

    def forward(self, vid, hidden, labels=None, validation=False):
        b, f, c, h, w = vid.shape  # vid has shape (batch_size * frames, channels, height, width)
        b = b * f
        slow_path = 2
        fast_path = 5
        b_slow = int (b / slow_path)
        b_fast = int (b / fast_path)
        # Slow and Fast paths (downsampling temporal dimension)
        slow_vid = vid[:, ::slow_path, :, :, :]  # Slow path (downsample by 8 along the last axis)
        fast_vid = vid[:, ::fast_path, :, :, :]  # Fast path (downsample by 2 along the last axis)
        slow_vid = slow_vid.view(-1,3,h,w)
        fast_vid = fast_vid.view(-1,3,h,w)
        # Slow Path
        slow_emb = self.embedding(slow_vid)  # (batch_size * frames, channels, height, width) #100,192,28,28
        slow_emb = rearrange(slow_emb, 'b c h w -> b (h w) c')  # Flatten to (batch_size, num_patches, hidden_size)
        slow_cls = repeat(self._cls, '() n d -> b n d', b=slow_emb.shape[0])  # Repeat class token
        _, _, c =slow_emb.shape
        slow_emb = torch.cat((slow_cls, slow_emb), dim=1)  # Concatenate class token
        slow_emb = slow_emb.reshape(b_slow // self.slow_length, self.slow_length, -1, c)  # Reshape for transformer
        slow_seq = repeat(self._seq, 'l () n d -> l b n d', b = b_slow // self.slow_length) #2,8,1,192->2,4,1,192
        hidden_slow = torch.cat((slow_seq, hidden), dim=2)  # Add sequence embeddings for slow path 2,4,785,192

        # Fast Path
        fast_emb = self.embedding(fast_vid)  # (batch_size * frames, channels, height, width)
        fast_emb = rearrange(fast_emb, 'b c h w -> b (h w) c')  # Flatten to (batch_size, num_patches, hidden_size)
        fast_cls = repeat(self._cls, '() n d -> b n d', b=fast_emb.shape[0])  # Repeat class token
        fast_emb = torch.cat((fast_cls, fast_emb), dim=1)  # Concatenate class token
        fast_emb = fast_emb.reshape(b_fast // self.fast_length, self.fast_length, -1, c)  # Reshape for transformer
        fast_seq = repeat(self._seq, 'l () n d -> l b n d', b = b_fast // self.fast_length)
        hidden_fast = torch.cat((fast_seq, hidden), dim=2)
        # hidden_fast = torch.cat((self._seq, hidden), dim=2)  # Add sequence embeddings for fast path
        # Frequency Path
        # 4, 50, 3, 224, 224
        vid_freq = vid.view(-1,3,h,w)
        # freq_feat = torch.fft.fft2(vid_freq)  # FFT on grayscale frames 4,3 224,224
        freq_feat = self.fdcf(vid_freq)
        # freq_feat = freq_feat.abs().mean(dim=(-2, -1))  # Take mean over spatial dims
        # transformer
        lp = int(self.length/(slow_path*fast_path))
        for k in range(lp):  #50/10=5 25,10 //5 //2
            for i in range(int(self.slow_length/lp)):  #25/5=5
                feat_slow, hidden_slow = self.transformer(slow_emb[:, i], hidden_slow, self.dropout(self.pos_embedding))
            for i in range(int(self.fast_length/lp)): #10/5=2
                feat_fast, hidden_fast = self.transformer(fast_emb[:, i], hidden_fast, self.dropout(self.pos_embedding))
            feat_freq = self.embedding(freq_feat)
            feat_freq = rearrange(feat_freq, 'b c h w -> b (h w) c')
            _,ch,_ = feat_freq.shape
            feat_freq = feat_freq.reshape(b // self.length, self.length*ch, c)
            if k % 2 == 0:
                freq_feat = self.fdcf(freq_feat)
            hidden = self.conv_fuse(hidden_slow + hidden_fast)
            feat_t = self.attention_weighted_fusion(feat_slow, feat_fast, feat_freq)
            feat_t = hidden[:, :, 0].transpose(0,1).contiguous()
            feat_t = feat_t.view(feat_t.size(0), -1)
            feat_st = torch.cat((feat_slow[:, 0], feat_t), dim=-1)

        cls, embedding = self.fc(self.dropout(self.to_cls_token(feat_st))) #4,5 此处也是分类结果

        return cls


class FC_class(nn.Module):
    def __init__(self, input_dim, class_num):
        super(FC_class, self).__init__()
        self.mlp = FeedForward(input_dim, input_dim * 4, 0.1)
        self.class_ = nn.Linear(input_dim, class_num)

    def forward(self, embedding):
        mlp_embedding = self.mlp(embedding)
        result=self.class_(mlp_embedding)
        return result, mlp_embedding


def RViT_P8_L16_112(**kwargs):
    input_size = 112
    patch_size = 8
    num_layers = 2
    num_classes = 400
    length = 16
    if 'num_classes' in kwargs:
        num_classes = kwargs['num_classes']

    return RViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        depth = num_layers,
        length = length,
        heads = 8,
        mlp_dim = patch_size ** 2 * 3 * 4,
        dropout = 0.1,
        emb_dropout = 0.1
    )

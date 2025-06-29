import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # 将图像分块并线性投影
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        return self.projection(x)

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        dropout=0.1
    ):
        super().__init__()
        
        # 图像分块并嵌入
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embedding.num_patches
        
        # 添加可学习的分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_ratio * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 获取批次大小
        batch_size = x.shape[0]
        
        # 图像分块嵌入
        x = self.patch_embedding(x)
        
        # 添加分类token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用CLS token进行分类
        x = x[:, 0]
        
        # MLP分类头
        return self.mlp_head(x)

# 使用示例
def train_vision_transformer():
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

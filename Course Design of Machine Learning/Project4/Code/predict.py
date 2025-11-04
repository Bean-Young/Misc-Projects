import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from model.Unet import UNet  # 或 UNetPlus
from model.UnetP import UNetPlus
# 设置路径
MODEL_PATH = '/home/yyz/Unet-ML/result/unetplus_model.pth'
TEST_IMG_PATH = '/home/yyz/Unet-ML/test.png'
GT_PATH = '/home/yyz/Unet-ML/pre0.png'
SAVE_PATH = '/home/yyz/Unet-ML/result/compare_test+.png'

# 超参数
NUM_CLASSES = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 颜色映射（与训练时保持一致）
color_dict = [
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128)
]

# 加载模型（如果是 RGB 输入）
model = UNetPlus(in_channels=3, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 图像预处理（RGB）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 加载测试图像（保持 RGB）
img = Image.open(TEST_IMG_PATH).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)

# 预测
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1)[0].cpu().numpy()

# 预测上色
pred_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
for cls in range(NUM_CLASSES):
    pred_rgb[pred == cls] = color_dict[cls]
pred_img = Image.fromarray(pred_rgb)
blended_pred = Image.blend(img, pred_img, alpha=0.7)

# 读取 GT 标签并上色
gt = Image.open(GT_PATH).convert('L').resize((512, 512), Image.NEAREST)
gt_np = np.array(gt)
gt_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
for cls in range(NUM_CLASSES):
    gt_rgb[gt_np == cls] = color_dict[cls]
gt_img = Image.fromarray(gt_rgb)
blended_gt = Image.blend(img, gt_img, alpha=0.7)

# 拼接显示 原图 | GT叠加图 | 预测叠加图
compare = Image.new('RGB', (512 * 3, 512))
compare.paste(img, (0, 0))
compare.paste(blended_gt, (512, 0))
compare.paste(blended_pred, (1024, 0))
compare.save(SAVE_PATH)

print(f"预测对比图已保存至: {SAVE_PATH}")
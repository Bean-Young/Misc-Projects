import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# 图像的预处理函数
def preprocess(img, image_shape):
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# 后处理函数
def postprocess(img):
    img = img.squeeze(0)  # 去掉 batch_size 维度
    img = torch.clamp(img.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]).to('cuda') + torch.tensor([0.485, 0.456, 0.406]).to('cuda'), 0, 1)
    return transforms.ToPILImage()(img.permute(2, 0, 1))  # 返回图片

# 加载预训练的VGG-19模型
pretrained_net = torchvision.models.vgg19(pretrained=True).features

# 将VGG模型移动到正确的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_net = pretrained_net.to(device)

# 使用VGG网络的某些层来提取内容和风格特征
content_layers = [21]  # 选择VGG的某层作为内容层
style_layers = [0, 5, 10, 19, 28]  # 选择VGG的某些层作为风格层

# 提取特征的函数
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(pretrained_net)):
        X = pretrained_net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 计算内容损失
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

# 计算风格损失
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 计算全变分损失
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 风格迁移的损失函数
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    content_loss_val = [content_loss(Y_hat, Y) for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    style_loss_val = [style_loss(Y_hat, gram_Y) for Y_hat, gram_Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_loss_val = tv_loss(X)
    return sum(content_loss_val + style_loss_val + [tv_loss_val])

# 读取内容图像和风格图像
content_img = Image.open('./Data/Content.jpg')  # 内容图像
style_img = Image.open('./Data/Style.jpg')  # 风格图像

# 设置图像形状和设备
image_shape = (300, 450)  # 图像大小
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预处理并提取内容和风格特征
content_X = preprocess(content_img, image_shape).to(device)
style_X = preprocess(style_img, image_shape).to(device)

# 使用VGG模型提取内容和风格特征
content_Y, _ = extract_features(content_X, content_layers, style_layers)
_, style_Y = extract_features(style_X, content_layers, style_layers)

# 计算风格的Gram矩阵
style_Y_gram = [gram(y) for y in style_Y]

# 初始化合成图像
generated_image = content_X.clone().requires_grad_(True).to(device)

# 定义优化器
optimizer = optim.Adam([generated_image], lr=0.3)

# 存储每个周期的损失
epoch_losses = []

# 训练模型
num_epochs = 500
for epoch in tqdm(range(num_epochs), desc="Training"):
    optimizer.zero_grad()
    contents_Y_hat, styles_Y_hat = extract_features(generated_image, content_layers, style_layers)
    loss = compute_loss(generated_image, contents_Y_hat, styles_Y_hat, content_Y, style_Y_gram)
    loss.backward()
    optimizer.step()

    epoch_losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 最终输出风格迁移后的图像
final_img = postprocess(generated_image)
final_img.save('./Result/style/final_styled_image.jpg')

# 绘制并保存训练曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('./Result/style/training_loss_curve.jpg')

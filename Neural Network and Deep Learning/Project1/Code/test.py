# test_pytorch.py
import torch
import torchvision

# 打印 PyTorch 和 torchvision 的版本
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# 检查是否支持 GPU
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready to use!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

# 创建一个简单的张量运算
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = x + y

print("Tensor x:")
print(x)
print("Tensor y:")
print(y)
print("x + y =")
print(z)
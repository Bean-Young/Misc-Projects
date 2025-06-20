import cv2
import numpy as np
import os
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_metrics(denoised, reference, name):
    p = psnr(reference, denoised, data_range=255)
    s = ssim(reference, denoised, channel_axis=-1, data_range=255)
    print(f"[{name}] PSNR: {p:.2f} dB | SSIM: {s:.4f}")


def apply_denoising_methods(noisy_img, noise_name, clean_img):
    save_dir = f'./Result/{noise_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 带噪图像
    cv2.imwrite(f'{save_dir}/noisy.jpg', cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    evaluate_metrics(noisy_img, clean_img, f'{noise_name} - Noisy Image')

    # 均值滤波
    mean_filtered = cv2.blur(noisy_img, (3, 3))
    cv2.imwrite(f'{save_dir}/mean_filtered.jpg', cv2.cvtColor(mean_filtered, cv2.COLOR_RGB2BGR))
    evaluate_metrics(mean_filtered, clean_img, f'{noise_name} - Mean Filter')

    # 中值滤波
    median_filtered = cv2.medianBlur(noisy_img, 3)
    cv2.imwrite(f'{save_dir}/median_filtered.jpg', cv2.cvtColor(median_filtered, cv2.COLOR_RGB2BGR))
    evaluate_metrics(median_filtered, clean_img, f'{noise_name} - Median Filter')

    # 卷积实现均值滤波
    kernel = np.ones((3, 3), np.float32) / 9
    conv_filtered = cv2.filter2D(noisy_img, -1, kernel)
    cv2.imwrite(f'{save_dir}/conv_filtered.jpg', cv2.cvtColor(conv_filtered, cv2.COLOR_RGB2BGR))
    evaluate_metrics(conv_filtered, clean_img, f'{noise_name} - Conv Mean Filter')

    # 双边滤波
    bilateral_filtered = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(f'{save_dir}/bilateral_filtered.jpg', cv2.cvtColor(bilateral_filtered, cv2.COLOR_RGB2BGR))
    evaluate_metrics(bilateral_filtered, clean_img, f'{noise_name} - Bilateral Filter')
    
    # 非局部均值滤波
    noisy_img_float = noisy_img.astype(np.float32) / 255.0
    sigma_est = np.mean(estimate_sigma(noisy_img_float, channel_axis=-1))
    nlm_denoised = denoise_nl_means(
        noisy_img_float,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=7,
        channel_axis=-1
    )
    nlm_denoised = (nlm_denoised * 255).astype(np.uint8)
    cv2.imwrite(f'{save_dir}/nlm_denoised.jpg', cv2.cvtColor(nlm_denoised, cv2.COLOR_RGB2BGR))
    evaluate_metrics(nlm_denoised, clean_img, f'{noise_name} - NLM Filter')

# 1. 加载图像
img = cv2.imread('./Data/Canon5D2_bag_mean.JPG')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 定义噪声种类
noise_types = {
    'salt_pepper': lambda img: random_noise(img, mode='s&p', amount=0.05),
    'gaussian': lambda img: random_noise(img, mode='gaussian', var=0.01),
    'poisson': lambda img: random_noise(img, mode='poisson'),
    'speckle': lambda img: random_noise(img, mode='speckle', var=0.05),
}

# 3.应用各种滤波器
for noise_name, noise_func in noise_types.items():
    noisy = noise_func(img_rgb)
    noisy = (noisy * 255).astype(np.uint8)
    apply_denoising_methods(noisy, noise_name, img_rgb)

# 4. 真实白噪声测试
real_noise_path = './Data/Canon5D2_bag_Real.JPG'
real_noisy = cv2.imread(real_noise_path)
real_noisy_rgb = cv2.cvtColor(real_noisy, cv2.COLOR_BGR2RGB)

apply_denoising_methods(real_noisy_rgb, 'real_white_noise', img_rgb)
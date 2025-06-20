import cv2
import numpy as np
import os

image_path = './Data/ISIC_0000003.jpg'
gt_path = './Data/ISIC_0000003_segmentation.png'

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gt = cv2.imread(gt_path, 0)
_, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)  

# 1. Canny 边缘检测
def canny_edge_detection(gray_img, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    return edges

edges = canny_edge_detection(gray)
cv2.imwrite('./Result/canny_edges.png', edges)

# 2. 二值化分割方法
def manual_threshold(gray_img, threshold=100):
    _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    return binary

def iterative_threshold(gray_img, epsilon=1):
    prev_T = gray_img.mean()
    while True:
        G1 = gray_img[gray_img > prev_T]
        G2 = gray_img[gray_img <= prev_T]
        if len(G1) == 0 or len(G2) == 0:
            break
        T = 0.5 * (G1.mean() + G2.mean())
        if abs(T - prev_T) < epsilon:
            break
        prev_T = T
    _, binary = cv2.threshold(gray_img, T, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    return binary, T

def otsu_threshold(gray_img):
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    return binary

# 3. 评估IoU 和 Dice
def compute_metrics(pred_mask, gt_mask):
    pred = (pred_mask == 255).astype(np.uint8)
    gt = (gt_mask == 255).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union != 0 else 0
    dice = 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 0
    return iou, dice

# Manual
manual_bin = manual_threshold(gray, threshold=100)
cv2.imwrite('./Result/manual_threshold_100.png', manual_bin)
iou_manual, dice_manual = compute_metrics(manual_bin, gt_bin)

# Iterative
iter_bin, iter_T = iterative_threshold(gray)
cv2.imwrite(f'./Result/iterative_threshold_{int(iter_T)}.png', iter_bin)
iou_iter, dice_iter = compute_metrics(iter_bin, gt_bin)

# OTSU
otsu_bin = otsu_threshold(gray)
cv2.imwrite('./Result/otsu_threshold.png', otsu_bin)
iou_otsu, dice_otsu = compute_metrics(otsu_bin, gt_bin)

# 形态学处理+(OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
otsu_morph = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel)   # 开运算去噪点
otsu_morph = cv2.morphologyEx(otsu_morph, cv2.MORPH_CLOSE, kernel)  # 闭运算填孔洞
cv2.imwrite('./Result/otsu_morph_open_close.png', otsu_morph)
iou_otsu_morph, dice_otsu_morph = compute_metrics(otsu_morph, gt_bin)

# 边缘平滑+(OTSU)
blurred = cv2.GaussianBlur(otsu_bin, (5, 5), 0)
_, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('./Result/otsu_morph_smooth.png', smooth_mask)
iou_otsu_smooth, dice_otsu_sooth = compute_metrics(smooth_mask, gt_bin)


print(f"Manual     - IoU: {iou_manual:.4f}, Dice: {dice_manual:.4f}")
print(f"Iterative  - IoU: {iou_iter:.4f}, Dice: {dice_iter:.4f} (T={iter_T:.2f})")
print(f"OTSU       - IoU: {iou_otsu:.4f}, Dice: {dice_otsu:.4f}")
print(f"OTSU+Morph - IoU: {iou_otsu_morph:.4f}, Dice: {dice_otsu_morph:.4f}")
print(f"OTSU+Smooth - IoU: {iou_otsu_smooth:.4f}, Dice: {dice_otsu_sooth:.4f}")
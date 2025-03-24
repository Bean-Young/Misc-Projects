import cv2
import imutils
import numpy as np
import math
from math import *
from scipy import ndimage
import matplotlib.pyplot as plt

def rotate_image(img_for_box_extraction_path):
    image_height = 1080
    image = cv2.imread(img_for_box_extraction_path)
    img = imutils.resize(image, height=image_height)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blur_gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    morhp_img = cv2.morphologyEx(blur_gray, cv2.MORPH_OPEN, kernel, (-1, -1))
    cv2.imwrite('./Figure/linesDetected.jpg', morhp_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    lines_img = cv2.dilate(morhp_img, kernel, iterations=1)
    cv2.imwrite("./Figure/lines_dilated.jpg", lines_img)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(lines_img, low_threshold, high_threshold)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=15, lines=np.array([]), minLineLength=50, maxLineGap=20)

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img, median_angle)
    print("Angle is {}".format(median_angle))
    cv2.imwrite('./Figure/rotated.jpg', img_rotated)
    return img_rotated


def warp_image(image_height, image):
    orig = image.copy()
    ratio = image.shape[0] / float(image_height)
    image = imutils.resize(image, height=image_height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0)

    # 保存对比图：Original和gray
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))  # matplotlib显示BGR转RGB
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(gray, cmap='gray')
    plt.title('gray')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig("./Figure/orig_gray_comparison.jpg")
    plt.close()

    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_cnt = None
    for c in contours:
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * epsilon, True)
        area = cv2.contourArea(c)

        if area < 2500:
            continue

        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        return -1, orig

    warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)

    # 保存对比图：处理后的Original和warped
    plt.figure(figsize=(12, 6))
    vis_image = image.copy()
    cv2.drawContours(vis_image, [screen_cnt], -1, (0, 255, 0), 2)
    for point in screen_cnt.reshape(4, 2):
        cv2.circle(vis_image, (point[0], point[1]), 5, (0, 0, 255), 4)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('Warped')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig("./Figure/vis_warped_comparison.jpg")
    plt.close()

    cv2.imwrite("./Figure/warped.jpg", warped)
    return warped


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def remove_line(warped_image):
    img_bin = imutils.resize(warped_image, height=1080)
    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
    (thresh, binary_src) = cv2.threshold(img_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite("./Figure/Image_bin_warp_invert.jpg", binary_src)
    kernel_length_horizontal = np.array(binary_src).shape[1] // 100
    kernel_length_vertical = np.array(binary_src).shape[0] // 30
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_vertical))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_horizontal, 1))
    img_temp1 = cv2.erode(binary_src, verticle_kernel, iterations=4)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=4)
    cv2.imwrite("./Figure/verticle_lines.jpg", verticle_lines_img)
    img_temp2 = cv2.erode(binary_src, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("./Figure/horizontal_lines.jpg", horizontal_lines_img)
    mask_img = verticle_lines_img + horizontal_lines_img
    binary_src = np.bitwise_xor(binary_src, mask_img)
    cv2.imwrite("./Figure/no_border_image.jpg", binary_src)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_erode = cv2.erode(binary_src, clean_kernel, iterations=1)
    binary_src = cv2.dilate(img_erode, clean_kernel, iterations=1)
    cv2.imwrite("./Figure/no_border_image_clean.jpg", binary_src)

if __name__ == '__main__':
    rotate_image(img_for_box_extraction_path='./Data/test.jpg')
    warped_image = warp_image(image_height=1080, image=cv2.imread('./Figure/rotated.jpg'))
    remove_line(warped_image)
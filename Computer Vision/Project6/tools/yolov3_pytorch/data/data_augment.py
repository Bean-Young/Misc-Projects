# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Data augmentation functions
"""
import math
import random
from typing import Any

import cv2
import numpy as np

__all__ = [
    "adjust_hsv", "cutout", "letterbox", "mixup", "random_affine",
]


def adjust_hsv(img: np.ndarray, h_gain: float = 0.5, s_gain: float = 0.5, v_gain: float = 0.5) -> np.ndarray:
    r"""Augment HSV channels of an image

    Args:
        img (np.ndarray): Image to augment
        h_gain (float): Hue gain
        s_gain (float): Saturation gain
        v_gain (float): Value gain

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> img = adjust_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5)

    Returns:
        np.ndarray: Augmented image
    """

    # Generate random gains for hue, saturation, and value channels
    random_gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1

    # Split the image into hue, saturation, and value channels
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    # Store the original data type of the image
    img_dtype = img.dtype

    # Create lookup tables for hue, saturation, and value channels
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * random_gains[0]) % 180).astype(img_dtype)
    lut_sat = np.clip(x * random_gains[1], 0, 255).astype(img_dtype)
    lut_val = np.clip(x * random_gains[2], 0, 255).astype(img_dtype)

    # Apply the lookup tables to the hue, saturation, and value channels
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(img_dtype)

    # Convert the image back to BGR color space
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img


def cutout(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    r"""Randomly add random square noise to the picture

    The aim is to improve generalization ability and robustness

    Args:
        img (np.ndarray): img from cv2.imread or np.ndarray
        labels (np.ndarray): img labels from YOLOv3 format

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> labels = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
        >>> labels = cutout(img, labels)

    Returns:
        np.ndarray: Augmented labels
    """

    img_h, img_w = img.shape[:2]

    def bbox_ioa(box1: np.ndarray, box2: np.ndarray):
        r"""Calculate the ratio of box1 and box2 intersecting area to box2 area

        Args:
            box1 (np.ndarray): box1 coordinates
            box2 (np.ndarray): box2 coordinates

        Returns:
            np.ndarray: Intersection over box2 area
        """

        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for scale in scales:
        mask_h = random.randint(1, int(img_h * scale))
        mask_w = random.randint(1, int(img_w * scale))

        # box
        xmin = max(0, random.randint(0, img_w) - mask_w // 2)
        ymin = max(0, random.randint(0, img_h) - mask_h // 2)
        xmax = min(img_w, xmin + mask_w)
        ymax = min(img_h, ymin + mask_h)

        # apply random color mask
        img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and scale > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            # intersection over area
            ioa = bbox_ioa(box, labels[:, 1:5])
            # remove >60% obscured labels
            labels = labels[ioa < 0.60]
    return labels

from typing import Any, Union, Tuple
def letterbox(
        img: np.ndarray,
        new_shape: int or tuple = (416, 416),
        color: tuple = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True
) -> tuple[
    Any,
    tuple[Union[float, Any], Union[float, Any]],
    tuple[Union[float, int, Any], Union[float, int, Any]]
]:
    """Resize image to a 32-pixel-multiple rectangle.

    Args:
        img (ndarray): Image to resize
        new_shape (int or tuple): Desired output shape of the image
        color (tuple): Color of the border
        auto (bool): Whether to choose the smaller dimension as the new shape
        scale_fill (bool): Whether to stretch the image to fill the new shape
        scaleup (bool): Whether to scale up the image if the image is smaller than the new shape

    Returns:
        ndarray: Resized image

    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def mixup(img1, labels, img2, labels2):
    r"""Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.
    
    Args:
        img1 (np.ndarray): Image to augment
        labels (np.ndarray): Image labels
        img2 (np.ndarray): Image to augment
        labels2 (np.ndarray): Image labels
    
    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> labels = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
        >>> img2 = cv2.imread("image2.jpg")
        >>> labels2 = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
        >>> img, labels = mixup(img, labels, img2, labels2)
    
    Returns:
        np.ndarray: Augmented image
        np.ndarray: Augmented labels
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    img1 = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return img1, labels


def random_affine(
        img: np.ndarray,
        targets: np.ndarray = (),
        degrees: float = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: float = 10,
        border: int = 0,
) -> tuple:
    r"""Random affine transformation of the image keeping center invariant

    Args:
        img (np.ndarray): Image to augment
        targets (tuple): Image targets
        degrees (float): Rotation angle
        translate (float): Translation
        scale (float): Scale
        shear (float): Shear
        border (int): Border

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> targets = np.array([[0, 0.1, 0.2, 0.3, 0.4]])
        >>> img, targets = random_affine(img, targets, degrees=10, translate=0.1, scale=0.1, shear=10, border=0)

    Returns:
        tuple: Augmented image and targets
    """
    img_h = img.shape[0] + border * 2
    img_w = img.shape[1] + border * 2

    # Rotation and Scale
    rotation_matrix = np.eye(3)
    rotation_angle = random.uniform(-degrees, degrees)
    rotation_scale = random.uniform(1 - scale, 1 + scale)
    rotation_matrix[:2] = cv2.getRotationMatrix2D(center=(img.shape[1] / 2, img.shape[0] / 2), angle=rotation_angle, scale=rotation_scale)

    # Translation
    translation_matrix = np.eye(3)
    translation_matrix[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    translation_matrix[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    shear_matrix[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    affine_matrix = shear_matrix @ translation_matrix @ rotation_matrix
    if (border != 0) or (affine_matrix != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, affine_matrix[:2], dsize=(img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    num_targets = len(targets)
    if num_targets:
        # warp points
        xy = np.ones((num_targets * 4, 3))
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(num_targets * 4, 2)
        xy = (xy @ affine_matrix.T)[:, :2].reshape(num_targets, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, num_targets).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, img_w)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, img_h)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        aspect_ratio = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 * scale + 1e-16) > 0.2) & (aspect_ratio < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

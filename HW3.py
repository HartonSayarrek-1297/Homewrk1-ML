import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
from skimage import data

img_bgr = cv2.imread("circus.jpg")

if img_bgr is None:
    print("Не удалось загрузить изображение")
else:
    print(img_bgr.shape)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H = img_bgr.shape[0]
W = img_bgr.shape[1]


def show_images(images, titles, rows=1, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for k, (img, title) in enumerate(zip(images, titles)):
        i = k//cols
        j = k%cols
        if len(img.shape) == 2:
            axes[i, j].imshow(img, cmap='gray')
        else:
            axes[i, j].imshow(img)
        axes[i, j].set_title(title)
        axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def stair(x):
    return 1 if x>0 else 0



weights=[1, -1, -1]
bias=-1
pixels = img_rgb.reshape(-1, 3).astype(np.float32)  # (H*W, 3)
z = pixels @ weights + bias
outputs = 1 / (1 + np.exp(-z))
neuron_mask = (outputs > 0.5).reshape(H, W).astype(np.uint8)

neuron_mask_res = cv2.bitwise_and(img_rgb, img_rgb, mask=neuron_mask)


img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_hsv = cv2.bitwise_or(mask1, mask2)

mask_hsv_res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_hsv)
mask_difference = cv2.bitwise_xor(mask_hsv, neuron_mask)

show_images([img_rgb, neuron_mask, neuron_mask_res, mask_hsv, mask_hsv_res, mask_difference], ["Ориг", "Нейронная маска", "Результат нейронной маски", "hsv-mask", "hsv-mask-result", "Разница"], 2, 3)

# print(f"Количество пикселей нейронной маски: {0}")
# print(f"Количество пикселей hsv-маски: {0}")
# print(f"Процент совпадения: {0}%")
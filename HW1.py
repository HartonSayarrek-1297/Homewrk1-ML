import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data

img_bgr = cv2.imread("pizza_furi.jpg")

if img_bgr is None:
    print("Не удалось загрузить изображение")
else:
    print(img_bgr.shape)
    
# plt.figure(figsize=(6,6))
# plt.imshow(img_bgr)
# plt.title("Загруженное")
# plt.axis('off')
# plt.show()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)



def show_images(images, titles, cols=3):
    fig, axes = plt.subplots(1, cols, figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
# show_images([img_bgr, img_rgb], ["Загруженное","Оригинал"], 2)

def sep_filter(img): # ручной рабоче-крестьянский фильтр сепии (☭)

    def convert(pixel):
        R_new = min(255, int(0.393 * pixel[0] + 0.769 * pixel[1] + 0.189 * pixel[2]))
        G_new = min(255, int(0.349 * pixel[0] + 0.686 * pixel[1] + 0.168 * pixel[2]))
        B_new = min(255, int(0.272 * pixel[0] + 0.534 * pixel[1] + 0.131 * pixel[2]))
        return (R_new, G_new, B_new)

    h = img.shape[0]
    w = img.shape[1]
    img_new = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            pix = img[i, j]
            img_new[i, j] = convert(pix)
        
    return img_new
    
img_remake = sep_filter(img_rgb)

show_images([img_rgb, img_remake], ["Жалкий оригинал", "Неповторимая пародия"], 2)
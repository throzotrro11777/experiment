import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_image(image, path):
    cv2.imwrite(path, image)

def add_noise(image, sigma):
    noisy_image = image.astype(np.float64) + np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
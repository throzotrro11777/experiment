import cv2
import numpy as np
from cbm3d import bm3d
from utils import load_image, save_image, add_noise, psnr
from config import Config


def main():
    config = Config()

    original_image = load_image('path/to/your/fabric_image.png')

    noisy_image = add_noise(original_image, config.sigma)

    cv2.imshow('Original Image', original_image)
    cv2.imshow('Noisy Image', noisy_image)

    denoised_image = bm3d(noisy_image, config)

    cv2.imshow('Denoised Image', denoised_image.astype(np.uint8))

    print(f'PSNR: {psnr(original_image, denoised_image)} dB')

    save_image(denoised_image.astype(np.uint8), 'path/to/save/denoised_image.png')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
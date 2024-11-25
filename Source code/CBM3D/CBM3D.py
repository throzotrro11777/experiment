import numpy as np
from scipy.fftpack import dct, idct
from config import Config


def BM3D_1st_step(sigma, image, config):
    N, M = image.shape
    patch_size = config.patch_size
    search_window = config.search_window
    max_match = config.max_match
    Threshold_Hard3D = config.Threshold_Hard3D
    Beta_Kaiser = config.Beta_Kaiser

    Basic_img = np.zeros((N, M), dtype=np.float64)
    Weight_sum = np.zeros((N, M), dtype=np.float64)

    for i in range(0, N - patch_size + 1, patch_size):
        for j in range(0, M - patch_size + 1, patch_size):
            block = image[i:i + patch_size, j:j + patch_size]
            search_x = max(0, i - search_window // 2)
            search_y = max(0, j - search_window // 2)
            search_x_end = min(N - patch_size, i + search_window // 2)
            search_y_end = min(M - patch_size, j + search_window // 2)

            similar_blocks = []
            for x in range(search_x, search_x_end + 1):
                for y in range(search_y, search_y_end + 1):
                    candidate_block = image[x:x + patch_size, y:y + patch_size]
                    if np.linalg.norm(block - candidate_block) < Threshold_Hard3D:
                        similar_blocks.append(candidate_block)

            if len(similar_blocks) > 0:
                similar_blocks = np.array(similar_blocks)
                transformed_blocks = dctn(similar_blocks, axes=(1, 2))
                thresholded_blocks = np.where(np.abs(transformed_blocks) < Threshold_Hard3D, 0, transformed_blocks)
                filtered_blocks = idctn(thresholded_blocks, axes=(1, 2))

                for k in range(len(filtered_blocks)):
                    x, y = i + (k % (search_x_end - search_x + 1)), j + (k // (search_x_end - search_x + 1))
                    Basic_img[x:x + patch_size, y:y + patch_size] += filtered_blocks[k]
                    Weight_sum[x:x + patch_size, y:y + patch_size] += 1

    Basic_img /= np.maximum(Weight_sum, 1)
    return Basic_img


def BM3D_2nd_step(sigma, image, basic, config):
    N, M = image.shape
    patch_size = config.patch_size
    search_window = config.search_window
    max_match = config.max_match
    Threshold_Hard3D = config.Threshold_Hard3D
    Beta_Kaiser = config.Beta_Kaiser

    Final_img = np.zeros((N, M), dtype=np.float64)
    Weight_sum = np.zeros((N, M), dtype=np.float64)

    for i in range(0, N - patch_size + 1, patch_size):
        for j in range(0, M - patch_size + 1, patch_size):
            block = basic[i:i + patch_size, j:j + patch_size]
            search_x = max(0, i - search_window // 2)
            search_y = max(0, j - search_window // 2)
            search_x_end = min(N - patch_size, i + search_window // 2)
            search_y_end = min(M - patch_size, j + search_window // 2)

            similar_blocks = []
            for x in range(search_x, search_x_end + 1):
                for y in range(search_y, search_y_end + 1):
                    candidate_block = basic[x:x + patch_size, y:y + patch_size]
                    if np.linalg.norm(block - candidate_block) < Threshold_Hard3D:
                        similar_blocks.append(candidate_block)

            if len(similar_blocks) > 0:
                similar_blocks = np.array(similar_blocks)
                transformed_blocks = dctn(similar_blocks, axes=(1, 2))
                weights = np.exp(-np.sum(transformed_blocks ** 2, axis=(1, 2)) / (2 * sigma ** 2))
                weighted_blocks = transformed_blocks * weights[:, None, None]
                filtered_blocks = idctn(weighted_blocks, axes=(1, 2))

                for k in range(len(filtered_blocks)):
                    x, y = i + (k % (search_x_end - search_x + 1)), j + (k // (search_x_end - search_x + 1))
                    Final_img[x:x + patch_size, y:y + patch_size] += filtered_blocks[k]
                    Weight_sum[x:x + patch_size, y:y + patch_size] += 1

    Final_img /= np.maximum(Weight_sum, 1)
    return Final_img


def dctn(array, axes=None):
    return dct(dct(array, axis=axes[0], norm='ortho'), axis=axes[1], norm='ortho')


def idctn(array, axes=None):
    return idct(idct(array, axis=axes[0], norm='ortho'), axis=axes[1], norm='ortho')


def bm3d(image, config):
    basic = BM3D_1st_step(config.sigma, image, config)
    final = BM3D_2nd_step(config.sigma, image, basic, config)
    return final
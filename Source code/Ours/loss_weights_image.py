import matplotlib.pyplot as plt
import numpy as np
from statistics import median


with open('loss_manual.txt', 'r') as file:
    lines = file.readlines()

fea_loss_weights = []
ssim_loss_weights = []
pix_loss_weights = []
perceptual_loss_weights = []


for line in lines:
    if 'fea_loss_weights' in line:
        weights_str = line.split(':')[1].strip().replace('[', '').replace(']', '').split(',')
        fea_loss_weights.extend([float(weight.strip()) for weight in weights_str])
    elif 'ssim_loss_weights' in line:
        weights_str = line.split(':')[1].strip().replace('[', '').replace(']', '').split(',')
        ssim_loss_weights.extend([float(weight.strip()) for weight in weights_str])
    elif 'pix_loss_weights' in line:
        weights_str = line.split(':')[1].strip().replace('[', '').replace(']', '').split(',')
        pix_loss_weights.extend([float(weight.strip()) for weight in weights_str])
    elif 'perceptual_loss_weights' in line:
        weights_str = line.split(':')[1].strip().replace('[', '').replace(']', '').split(',')
        perceptual_loss_weights.extend([float(weight.strip()) for weight in weights_str])


batch_indices = range(1, len(fea_loss_weights) + 1)


plt.figure(figsize=(14, 7))


# plt.plot(batch_indices, fea_loss_weights, label='fea_loss_weights', marker='o')
plt.plot(batch_indices, fea_loss_weights, label='fea_loss_weights')
plt.axhline(y=median(fea_loss_weights), color='r', linestyle='--', label=f'Median fea_loss_weights: {median(fea_loss_weights):.2f}')

# plt.plot(batch_indices, ssim_loss_weights, label='fea_loss_weights', marker='*')
plt.plot(batch_indices, ssim_loss_weights, label='fea_loss_weights')
plt.axhline(y=median(ssim_loss_weights), color='y', linestyle='--', label=f'Median fea_loss_weights: {median(ssim_loss_weights):.2f}')


# plt.plot(batch_indices, pix_loss_weights, label='pix_loss_weights', marker='s')
plt.plot(batch_indices, pix_loss_weights, label='pix_loss_weights')
plt.axhline(y=median(pix_loss_weights), color='g', linestyle='--', label=f'Median pix_loss_weights: {median(pix_loss_weights):.2f}')

# plt.plot(batch_indices, perceptual_loss_weights, label='perceptual_loss_weights', marker='^')
plt.plot(batch_indices, perceptual_loss_weights, label='perceptual_loss_weights')
plt.axhline(y=median(perceptual_loss_weights), color='b', linestyle='--', label=f'Median perceptual_loss_weights: {median(perceptual_loss_weights):.2f}')


plt.title('Loss Weights over Batches with Median Lines')
plt.xlabel('Batch')
plt.ylabel('Weight Value')
plt.legend()


plt.xlim(left=0)
plt.ylim(0.1, 0.75)

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)


plt.show()
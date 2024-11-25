import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.vgg = vgg19(pretrained=True).features[:35].eval().requires_grad_(False)
        self.vgg = self.vgg.cuda() if torch.cuda.is_available() else self.vgg

    def perceptual_loss(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.mse_loss(x_features, y_features)

    def gradient_loss(self, x, y):
        x_diff_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        x_diff_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        y_diff_x = y[:, :, :, 1:] - y[:, :, :, :-1]
        y_diff_y = y[:, :, 1:, :] - y[:, :, :-1, :]
        return self.mse_loss(x_diff_x, y_diff_x) + self.mse_loss(x_diff_y, y_diff_y)

    def forward(self, x, y):
        mse = self.mse_loss(x, y)
        perceptual = self.perceptual_loss(x, y)
        ssim = 1 - self.ssim_loss(x, y)  # SSIM is between 0 and 1, so we use 1 - SSIM
        gradient = self.gradient_loss(x, y)
        return self.alpha * mse + self.beta * perceptual + self.gamma * ssim + self.delta * gradient

def get_loss_function(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    return CustomLoss(alpha, beta, gamma, delta)
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def calculate_psnr(clean_images, denoised_images):
    psnr = PeakSignalNoiseRatio().to(clean_images.device)
    return psnr(denoised_images, clean_images)

def calculate_ssim(clean_images, denoised_images):
    ssim = StructuralSimilarityIndexMeasure().to(clean_images.device)
    return ssim(denoised_images, clean_images)

def evaluate_metrics(clean_images, denoised_images):
    psnr_value = calculate_psnr(clean_images, denoised_images)
    ssim_value = calculate_ssim(clean_images, denoised_images)
    return psnr_value, ssim_value

if __name__ == "__main__":
    clean_images = torch.randn(4, 3, 512, 512).cuda()
    denoised_images = torch.randn(4, 3, 512, 512).cuda()

    psnr_value, ssim_value = evaluate_metrics(clean_images, denoised_images)
    print(f'PSNR: {psnr_value:.4f}')
    print(f'SSIM: {ssim_value:.4f}')
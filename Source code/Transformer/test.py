import torch
import torch.nn as nn
from torchvision.models import vgg19
from data_loader import get_data_loaders
from loss import get_loss_function
from train import DenoisingTransformer
from metrics import evaluate_metrics

def test_model(model_path, test_dir, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingTransformer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = get_loss_function()
    _, test_loader = get_data_loaders(None, test_dir, batch_size)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for noisy_images, clean_images in test_loader:
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            total_loss += loss.item()

            psnr_value, ssim_value = evaluate_metrics(clean_images, outputs)
            total_psnr += psnr_value
            total_ssim += ssim_value

    avg_loss = total_loss / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')

if __name__ == "__main__":
    model_path = 'denoising_transformer.pth'
    test_dir = 'test_data'
    test_model(model_path, test_dir)
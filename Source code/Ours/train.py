import math
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import loss
from data import *
from model import *


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.Tensor

    epochs = 2000
    sample_interval = 5
    last_epoch = 0

    data_process_steps = [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    train_dir = "./data"

    train_data = DataLoader(
        img_data(train_dir, transforms_=data_process_steps),
        batch_size=1,
        num_workers=4,
        shuffle=True
    )

    gen = GenUnet().to(device)
    dis = MultiScaleDiscriminator().to(device)
    vgg = VGG19_fea().to(device)

    adv_loss = nn.BCELoss()
    pix_loss = nn.MSELoss()
    fea_loss = nn.MSELoss()
    ssim_loss = loss.SSIM()

    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.05, 0.999))
    opt_d = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.05, 0.999))

    def PSNR(img1, img2):
        MSE = nn.MSELoss()
        psnr = 10 * math.log10((255 ** 2) / MSE(img1, img2).item())
        return psnr

    def sample_images(batches_done, train_data, Tensor):
        train_iterator = iter(train_data)
        batch_index_to_load = 2
        for _ in range(batch_index_to_load):
            try:
                imgs = next(train_iterator)
            except StopIteration:
                return

        print("Groundtruth save Sample image path: ", imgs["img_path_ground"])
        print("Noise save Sample image path: ", imgs["img_path_noise"])

        gen.eval()

        noise = Variable(imgs["noise"].type(Tensor))
        dis_noise = gen(noise)
        ground = Variable(imgs["groundtruth"].type(Tensor))

        ssim_loss_value = ssim_loss(dis_noise, ground).item()
        psnr_loss_value = PSNR(dis_noise, ground)

        img_noise = make_grid(noise, nrow=5, normalize=True)
        img_dis = make_grid(dis_noise, nrow=5, normalize=True)
        img_ground = make_grid(ground, nrow=5, normalize=True)

        image_grid = torch.cat((img_noise, img_dis, img_ground), 1)
        save_image(image_grid, f"train_image/{batches_done},SSIM={ssim_loss_value:.3f},PSNR={psnr_loss_value:.3f}.png",
                   normalize=False)

    def vgg_loss(gen_features, target_features):
        return torch.mean(torch.abs(gen_features - target_features))

    def load_model_state(model, optimizer, epoch, model_name):
        model_path = f"saved_model/{model_name}_{epoch}.pth"
        optimizer_path = f"saved_model/opt_{model_name}_{epoch}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"未找到{model_path}预训练模型")

        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            print(f"未找到{optimizer_path}预训练模型")

    load_model_state(gen, opt_g, last_epoch - 1, "Gen")
    load_model_state(dis, opt_d, last_epoch - 1, "Dis")

    gg = []
    dg = []
    weights = {'g_loss': [], 'img_loss': [], 'fea_loss': [], 'ssim_loss': [], 'pix_loss': [], 'perceptual_loss': []}
    times = {'batch_times': [], 'epoch_times': []}

    def compute_adv_loss(outputs, valid_list):
        return sum([adv_loss(output, valid) for output, valid in zip(outputs, valid_list)]) / len(outputs)

    def compute_loss(outputs, valid_list, fake_list):
        real_loss = sum([adv_loss(output, valid) for output, valid in zip(outputs, valid_list)])
        fake_loss = sum([adv_loss(output, fake) for output, fake in zip(outputs, fake_list)])
        return real_loss, fake_loss

    def get_output_shape(discriminator, image_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape).to(device)
            output = discriminator(dummy_input)
        return output[0].shape

    groundtruth_shape = (3, 256, 256)
    dis_output_shape = get_output_shape(dis.discriminators[0], groundtruth_shape)

    def calculate_gradients_and_weights(losses, gen, alpha=2, beta=2.5):
        loss_gradients = [torch.abs(torch.autograd.grad(loss_item, gen.parameters(), retain_graph=True)[0]) for
                          loss_item in losses]
        gradient_sums = [torch.sum(grad).item() for grad in loss_gradients]
        gradient_weights = [alpha * abs_gradient_sum / sum(gradient_sums) + beta for abs_gradient_sum in gradient_sums]
        weighted_losses = [grad_weight * loss_item for grad_weight, loss_item in zip(gradient_weights, losses)]
        return sum(weighted_losses), gradient_weights

    def dynamic_weight_adjustment(epoch, batch_idx, total_batches, alpha=1, beta=0.1):
        progress = (epoch * total_batches + batch_idx) / (epochs * total_batches)
        weight_adv = max(1 - progress, 0.1)
        weight_pix_fea_perceptual = 2 - weight_adv
        return weight_adv, weight_pix_fea_perceptual

    def save_model_and_log(epoch, gen, dis, opt_g, opt_d, gg, dg, weights, times):
        torch.save(gen.state_dict(), f"saved_model/Gen_{epoch}.pth")
        torch.save(dis.state_dict(), f"saved_model/Dis_{epoch}.pth")
        torch.save(opt_g.state_dict(), f"saved_model/opt_g_{epoch}.pth")
        torch.save(opt_d.state_dict(), f"saved_model/opt_d_{epoch}.pth")

        with open("loss.txt", "w") as f:
            for g, d in zip(gg, dg):
                f.write(f"G_loss: {g}, D_loss: {d}\n")
            for weight_key in weights.keys():
                f.write(f"{weight_key}_weights: {weights[weight_key]}\n")

        with open("times.txt", "w") as f:
            f.write("Batch times:\n")
            for batch_time in times['batch_times']:
                f.write(f"{batch_time}\n")
            f.write("Epoch times:\n")
            for epoch_time in times['epoch_times']:
                f.write(f"{epoch_time}\n")

    def compute_discriminator_loss(dis, groundtruth, dis_noise, Tensor):
        real_outputs = dis(groundtruth)
        fake_outputs = dis(dis_noise.detach())

        valid_outputs = [Variable(Tensor(np.ones(output.size())), requires_grad=False) for output in real_outputs]
        fake_outputs_vars = [Variable(Tensor(np.zeros(output.size())), requires_grad=False) for output in fake_outputs]

        real_loss, _ = compute_loss(real_outputs, valid_outputs, fake_outputs_vars)
        _, fake_loss = compute_loss(fake_outputs, valid_outputs, fake_outputs_vars)
        return (real_loss + fake_loss) / 2

    total_batches = len(train_data)

    for epoch in range(last_epoch, epochs + last_epoch):
        epoch_start_time = time.time()
        g_losses = []
        d_losses = []

        for i, batch in enumerate(train_data):
            batch_start_time = time.time()

            groundtruth = Variable(batch["groundtruth"].type(Tensor))
            noise = Variable(batch["noise"].type(Tensor))

            img_paths_ground = batch["img_path_ground"]
            img_paths_noise = batch["img_path_noise"]

            print(f"Training on groundtruth: {img_paths_ground}    noise: {img_paths_noise}")

            opt_g.zero_grad()

            dis_noise = gen(noise)
            gen_outputs = dis(dis_noise)

            valid_outputs = [Variable(Tensor(np.ones(output.size())), requires_grad=False) for output in gen_outputs]

            g_loss = compute_adv_loss(gen_outputs, valid_outputs)

            ssim = ssim_loss(dis_noise, groundtruth)
            ssim_ = abs(ssim - 1)

            pix = pix_loss(dis_noise, groundtruth)

            img_loss = ssim_ + pix

            dis_fea = vgg(dis_noise)
            ground_fea = vgg(groundtruth)
            fea = fea_loss(dis_fea, ground_fea)

            perceptual_loss = vgg_loss(dis_fea, ground_fea)

            weight_adv, weight_pix_fea_perceptual = dynamic_weight_adjustment(epoch, i, total_batches)

            total_adaptive_loss, gradient_weights = calculate_gradients_and_weights(
                [weight_adv * g_loss, weight_pix_fea_perceptual * img_loss, weight_pix_fea_perceptual * fea,
                 weight_pix_fea_perceptual * ssim_, weight_pix_fea_perceptual * pix,
                 weight_pix_fea_perceptual * perceptual_loss], gen
            )

            g_losses.append(total_adaptive_loss.item())

            total_adaptive_loss.backward()
            opt_g.step()

            opt_d.zero_grad()

            d_loss = compute_discriminator_loss(dis, groundtruth, dis_noise, Tensor)
            d_losses.append(d_loss.item())

            d_loss.backward()
            opt_d.step()

            batch_end_time = time.time()
            times['batch_times'].append(batch_end_time - batch_start_time)

            batches_done = epoch * len(train_data) + i
            print(
                f"Epoch: {epoch}/{last_epoch + epochs - 1}, Batch: {i}/{len(train_data)}, D loss: {d_loss.item():.4f}, "
                f"G loss: {g_loss.item():.4f}, img loss: {img_loss.item():.4f}, feature loss: {fea.item():.4f}, "
                f"ssim: {ssim.item():.4f}, pix: {pix.item():.4f}, perceptual_loss: {perceptual_loss.item():.4f}, "
                f"total G: {total_adaptive_loss.item():.4f}"
            )

            if batches_done % sample_interval == 0:
                sample_images(batches_done, train_data, Tensor)

            for key, weight in zip(weights.keys(), gradient_weights):
                weights[key].append(weight)

        epoch_end_time = time.time()
        times['epoch_times'].append(epoch_end_time - epoch_start_time)

        if g_losses:
            gg.append(sum(g_losses) / len(g_losses))
        if d_losses:
            dg.append(sum(d_losses) / len(d_losses))

        save_model_and_log(epoch, gen, dis, opt_g, opt_d, gg, dg, weights, times)

    # 计算固定权重
    fixed_weights = {key: sum(values) / len(values) for key, values in weights.items()}

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(gg)), gg, label='Generator Loss')
    plt.plot(range(len(dg)), dg, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for key in weights.keys():
        plt.plot(range(len(weights[key])), weights[key], label=f'{key} Weight')
    for key, fixed_weight in fixed_weights.items():
        plt.axhline(y=fixed_weight, color='r', linestyle='--', label=f'Fixed {key} Weight')
    plt.xlabel('Batch')
    plt.ylabel('Weight')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_progress.png')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(times['batch_times'])), times['batch_times'], label='Batch Times')
    plt.xlabel('Batch')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(times['epoch_times'])), times['epoch_times'], label='Epoch Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_times.png')

    with open("fixed_weights.txt", "w") as f:
        for key, fixed_weight in fixed_weights.items():
            f.write(f"Fixed {key} Weight: {fixed_weight}\n")


if __name__ == "__main__":
    train()

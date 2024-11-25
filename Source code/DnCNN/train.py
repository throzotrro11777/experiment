import torch
import torch.optim as optim
from data_loader import get_data_loaders
from model import get_model
from loss import get_loss_function

def train_model(train_dir, test_dir, num_epochs=100, batch_size=4, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    criterion = get_loss_function(alpha=1.0, beta=0.1, gamma=0.1, delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for noisy_images, clean_images in train_loader:
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), 'dncnn_denoiser.pth')

if __name__ == "__main__":
    train_dir = 'train_data'
    test_dir = 'test_data'
    train_model(train_dir, test_dir)
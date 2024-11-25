import torch
import torch.optim as optim
from torchvision.models import vgg19
from data_loader import get_data_loaders
from loss import get_loss_function

class DenoisingTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, kernel_size=1),
            nn.Tanh()
        )
        self.patch_size = patch_size

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(x.shape[0], -1, self.patch_size, self.patch_size)
        x = self.decoder(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2).transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=False, drop_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

def train_model(train_dir, test_dir, num_epochs=10, batch_size=4, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingTransformer().to(device)
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

    torch.save(model.state_dict(), 'denoising_transformer.pth')

if __name__ == "__main__":
    train_dir = 'train_data'
    test_dir = 'test_data'
    train_model(train_dir, test_dir)
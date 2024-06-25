import os
import time
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1_miu = nn.Linear(input_size, hidden_size)
        self.fc2_miu = nn.Linear(hidden_size, output_size)

        self.fc1_sigma = nn.Linear(input_size, hidden_size)
        self.fc2_sigma = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_miu = self.fc1_miu(x)
        x_miu = self.relu(x_miu)
        x_miu = self.fc2_miu(x_miu)

        x_sigma = self.fc1_sigma(x)
        x_sigma = self.relu(x_sigma)
        x_sigma = self.fc2_sigma(x_sigma)

        return x_miu, x_sigma

class VAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.miu = MLP(input_size, hidden_size, latent_size)
        self.sigma = MLP(input_size, hidden_size, latent_size)

    def forward(self, x):
        x_miu, x_sigma = self.miu(x)
        return x_miu, x_sigma

class VAE_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, observable_size):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, observable_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Assuming normalized output

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, observable_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, observable_size)

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        recon_x = self.decoder(z)
        return recon_x, mu, sigma

    def loss_function(self, recon_x, x, mu, sigma):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return BCE + KLD


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.img_labels[idx]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_dataloader(img_dir, batch_size=32, shuffle=True, num_workers=1):
    dataset = ImageDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader
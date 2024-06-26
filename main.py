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
        BCE = torch.nn.functional.MSELoss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return BCE + KLD


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB') ## What's YCbCr: luminance + blue and red color
    return img


def tens2PIL_display(tensor):

    to_pil = transforms.ToPILImage()
    pil_img = to_pil(tensor)
    
    pil_img.show()
    
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class DatasetFromFolder(Dataset):
    def __init__(self, input_dir, target_dir=None, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.input_filenames = [join(input_dir, x) for x in sorted(listdir(input_dir)) if is_image_file(x)]

        if target_dir is not None:
            self.target_filenames = [join(target_dir, x) for x in sorted(listdir(target_dir)) if is_image_file(x)]
        else:
            self.target_filenames = self.input_filenames.copy()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.input_filenames[index])
        target = load_img(self.target_filenames[index])

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.input_filenames)


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class Transformations(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = self.totensor(img)

        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return img

    def reverse_transformation(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        img = self.topil(tensor)
        return img


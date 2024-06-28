import os
import time
from os import listdir
from os.path import join
import argparse
import gc
from PIL import Image
import matplotlib.pyplot as plt
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.miu = MLP(input_size=input_size, hidden_size=hidden_size, output_size=latent_size)
        self.sigma = MLP(input_size=input_size, hidden_size=hidden_size, output_size=latent_size)

    def forward(self, x): 
        x_miu, x_sigma = self.miu(x), self.sigma(x)

        return x_miu, x_sigma

class VAE_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, observable_size):
        super(VAE_Decoder, self).__init__()

        self.decode_mu    = MLP(input_size=input_size, hidden_size=hidden_size, output_size=observable_size)
        self.decode_sigma = MLP(input_size=input_size, hidden_size=hidden_size, output_size=observable_size)

    #### todo: Comparative study of having decoder as a distribution or point estimation
    def forward(self, x): #[B,C,Z]
        x_mu    = self.decode_mu(x) 
        x_sigma = self.decode_sigma(x)

        #x = self.decode(x)
        return F.sigmoid(x_mu), x_sigma


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, observable_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, observable_size)
        self.pi = torch.tensor(torch.pi, device=device)

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        mu, sigma = self.encoder(x) 
        z = self.reparameterize(mu, sigma) 
        dec_mu, dec_sigma = self.decoder(z)

        return mu, sigma, dec_mu, dec_sigma 

    def loss_function(self, x, mu, sigma, mu_recon, sigma_recon):
        x_recon = self.reparameterize(mu_recon, sigma_recon)
        BCE = F.mse_loss(x_recon, x, reduction='sum')

        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return BCE + KLD



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        return img
    except OSError as e:
        print(f"Warning corrupted image {filepath}: {e}")
        return None


def tens2PIL_display(tensor):

    to_pil = transforms.ToPILImage()
    pil_img = to_pil(tensor)

    pil_img.show()

    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4)


class DatasetFromFolder(Dataset):
    def __init__(self, input_dir, input_transform=None):
        super(DatasetFromFolder, self).__init__()
        #self.input_filenames, self.invalid = self._get_valid_images(input_dir)
        self.input_transform = input_transform
        self.data = self._load_all_images(input_dir)

    def _load_all_images(self, input_dir):
        data = []
        for filename in listdir(input_dir):
            if is_image_file(filename):
                filepath = join(input_dir, filename)
                img = load_img(filepath)
                if img is not None:
                    if self.input_transform:
                        img = self.input_transform(img)
                    data.append(img)
        return data


    def __getitem__(self, index):
        input_img  = self.data[index]
        target_img = self.data[index]

        C, H, W = input_img.shape
        input_img  = input_img.view(C * H * W)
        target_img = target_img.view(C * H * W)


        return input_img, target_img

    def __len__(self):
        return len(self.data)


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_data(input_dir, resize_size):
    output = []
    resize = transforms.Resize(resize_size)
    totensor = transforms.ToTensor()

    filenames = [join(input_dir, x) for x in sorted(listdir(input_dir)) if is_image_file(x)]
    for idx in range(len(filenames)):
        img = load_img(filenames[idx])

        if img is not None:
            img = resize(img)
            if isinstance(img, Image.Image):
                img = totensor(img)
            output.append(img)

    output = torch.stack(output)
    return output


class Transformations(object):
    def __init__(self, mean, std, resize_size=None):
        self.mean = mean
        self.std  = std
        self.totensor = transforms.ToTensor()
        self.topil    = transforms.ToPILImage()
        self.resize   = transforms.Resize(resize_size)

    def __call__(self, img):

        img = self.resize(img)
        if isinstance(img, Image.Image):
            img = self.totensor(img)

        #for t, m, s in zip(img, self.mean, self.std):
            #t.sub_(m).div_(s)

        return img

    def reverse_transformation(self, tensor):
        #for t, m, s in zip(tensor, self.mean, self.std):
            #t.mul_(s).add_(m)

        img = self.topil(tensor)
        return img
    

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)
        B,C,H,W = data.shape
        data = data.view(B,C*H*W)
        optimizer.zero_grad()
        mu, sigma, mu_dec, sigma_dec = model(data)
        loss = model.loss_function(data, mu, sigma, mu_dec, sigma_dec)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_dataloader.dataset)))

"""
FOR THE MNIST DATASET
"""
dataset = datasets.MNIST('./Datasets/MNIST', train=True, download=True, transform=transforms.ToTensor())
topil  = transforms.ToPILImage()


train_size = int(0.7 * len(dataset))
remaining_size = len(dataset) - train_size
validation_size = remaining_size // 2
test_size = remaining_size - validation_size

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_dataloader = create_dataloader(train_dataset, batch_size=128, shuffle=True)
validation_dataloader = create_dataloader(validation_dataset, batch_size=16, shuffle=False)
test_dataloader = create_dataloader(test_dataset, batch_size=16, shuffle=False)


input_dim = 1*28*28
hidden_dim = 400
latent_dim =  20
observable_size = 1*28*28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VAE(input_dim, hidden_dim, latent_dim, observable_size)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


    
for epoch in range(1, 1000):
    train(epoch)

## Random sampling from https://github.com/pytorch/examples/tree/main/vae
with torch.no_grad():
    sample = torch.randn(64, 20).to(device)
    sample = model.decoder(sample.unsqueeze(0))
    sample = model.reparameterize(sample[0].cpu(), sample[1].cpu())
    save_image(sample.view(64, 1, 28, 28),(f'./continuous.png'))


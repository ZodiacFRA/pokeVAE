#!/usr/bin/env python3
from pprint import pprint

import torch
import torchvision
from torchvision import utils, transforms
import matplotlib.pyplot as plt
import numpy as np

from GLOBALS import *
from VAE import VAE
from PokemonDataset import PokemonDataset
from utils import *


def get_sample(n, start, end):
    props = []
    xl = np.arange(start, end, (end - start)/n)
    yl = np.arange(start, end, (end - start)/n)
    for x in xl:
        for y in yl:
            props.append((x, y))
    return torch.FloatTensor(props).to(DEVICE)


def train(epoch, warmup_factor, model, optimizer, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(DEVICE)
        data = data.transpose(1, 3)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, warmup_factor)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.item() / len(data)))

    print(f"====> Epoch: {epoch} Average loss: {(train_loss / len(dataloader.dataset)):.4f}, Warmup Factor: {warmup_factor}")


if __name__ == '__main__':
    face_dataset = PokemonDataset(
        csv_file='./pokemons/pokemon.csv',
        root_dir='./pokemons/images',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
        ]))
    # fig = plt.figure()
    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]
    #
    #     print(i, sample.shape)
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.axis('off')
    #     plt.imshow(sample)
    #
    #     if i == 3:
    #         plt.show()
    #         break
    # exit()

    train_dataloader = torch.utils.data.DataLoader(face_dataset, batch_size=BATCH_SIZE)

    model = VAE(3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create 2D representation by varying the value of each latent variable
    # n = 40
    # big_sample = get_sample(n, -5, 5)
    # small_sample = get_sample(n, -1, 1)

    # n=1
    # one_shot = torch.FloatTensor(((0, 0))).to(DEVICE)

    for epoch in range(0, EPOCHS):
        warmup_factor = min(1, epoch / WARMUP_TIME)
        with torch.no_grad():
            # res = model.decode(one_shot).cpu()
            # res = model.decode(big_sample).cpu()
            res = model.sample(10)
            torchvision.utils.save_image(res.view(10, 3, 64, 64).cpu(), f"./results/reconstruction_{epoch}_{warmup_factor}_big.png", nrow=10)
            # res = model.decode(small_sample).cpu()
            # torchvision.utils.save_image(res.view(n*n, 3, 64, 64).cpu(), f"./results/reconstruction_{epoch}_{warmup_factor}_small.png", nrow=n)

        train(epoch, warmup_factor, model, optimizer, train_dataloader)

#!/usr/bin/env python3
import sys
import time
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


def train(epoch, warmup_factor, model, optimizer, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(DEVICE)
        # data = data.transpose(1, 3)
        # print("data train", data.shape)
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


def draw_dataset_sample(train_dataset):
    fig = plt.figure()
    fig.patch.set_facecolor('#222222')
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        print(i, sample.shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.axis('off')
        plt.imshow(sample)
        if i == 3:
            plt.show()
            break


def predict(model, sample, epoch, n):
    with torch.no_grad():
        res = model.decode(sample).cpu()
        torchvision.utils.save_image(res.view(n*n, 1, 64, 64).cpu(), f"./results/reconstruction_{epoch}.png", nrow=n)


def get_sample(n, start, end):
    props = []
    xl = np.arange(start[0], end[0], (end[0] - start[0])/n)
    yl = np.arange(start[1], end[1], (end[1] - start[1])/n)
    for x in xl:
        for y in yl:
            props.append((x, y))
    return torch.FloatTensor(props).to(DEVICE)


if __name__ == '__main__':
    train_dataset = PokemonDataset(
        csv_file='./pokemons/pokemon.csv',
        root_dir='./pokemons/images',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
        ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    model = VAE(64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    # Create 2D representation by varying the value of each latent variable
    n = 40
    big_sample = get_sample(n, (-7, 7), (-2, 7))
    # os_sample = model.sample(10)

    if len(sys.argv) == 1:
        print("training")
        for epoch in range(0, EPOCHS):
            warmup_factor = min(1, epoch / WARMUP_TIME)
            if epoch % LOG_INTERVAL == 0:
                predict(model, big_sample, epoch, n)
            train(epoch, warmup_factor, model, optimizer, train_dataloader)
        torch.save(model.state_dict(), f'./{time.time()}.pth')
    else:
        print("testing")
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        predict(model, get_sample(n, (-7, 7), (-2, 7)), '0', n)
        predict(model, get_sample(n, (-2, 7), (-7, 7)), '0', n)

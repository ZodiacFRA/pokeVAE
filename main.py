#!/usr/bin/env python3
import sys
import time

import torch
import torchvision

from GLOBALS import *
from cVAE import cVAE
from VAE import VAE
from PokemonDataset import PokemonDataset
from utils import *


def train(epoch, warmup_factor, model, optimizer, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(DEVICE)

        data = data.transpose(1, 3)  # Needed for conv layers

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, warmup_factor)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f"====> Epoch: {epoch} Average loss: {(train_loss / len(dataloader.dataset)):.4f}, Warmup Factor: {warmup_factor}")


if __name__ == '__main__':
    image_size = 64
    channels_nbr = 3
    train_dataset = PokemonDataset(
        draw_samples=False,
        csv_file='./pokemons/pokemon.csv',
        root_dir='./pokemons/images',
        transform=torchvision.transforms.Compose([
            Rescale(image_size),
            SetChannels(image_size, channels_nbr, pad=True),
            ToTensor()
        ]))

    # train_dataset.draw_dataset_sample()
    # exit()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    model = cVAE(image_size, channels_nbr).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    n_samples = 40
    big_sample = get_sample(n_samples, (-5, -5), (5, 5))
    # rand_sample = torch.randn(n_samples*n_samples, LATENT_SPACE_SIZE).to(DEVICE)

    if len(sys.argv) == 1:
        print('='*50, "Training")
        for epoch in range(0, EPOCHS):
            warmup_factor = min(1, (epoch + 1) / WARMUP_TIME)
            train(epoch, warmup_factor, model, optimizer, train_dataloader)
            if epoch % LOG_INTERVAL == 0:
                # Check with a prediction
                with torch.no_grad():
                    res = model.decode(big_sample).cpu()
                # Save preview
                torchvision.utils.save_image(
                    res.view(n_samples*n_samples, channels_nbr, image_size, image_size).cpu(),
                    f"./results/reconstruction_{epoch}.png",
                    nrow=n_samples
                )

        torch.save(model.state_dict(), f'./{time.time()}.pth')
    elif len(sys.argv) == 2:
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        for epoch in range(0, sys.argv[2]):
            warmup_factor = min(1, (epoch + 1) / WARMUP_TIME)
            train(epoch, warmup_factor, model, optimizer, train_dataloader)
            if epoch % LOG_INTERVAL == 0:
                # Check with a prediction
                with torch.no_grad():
                    res = model.decode(big_sample).cpu()
                # Save preview
                torchvision.utils.save_image(
                    res.view(n_samples*n_samples, channels_nbr, image_size, image_size).cpu(),
                    f"./results/reconstruction_{epoch}.png",
                    nrow=n_samples
                )

        torch.save(model.state_dict(), f'./{time.time()}_RESUMED.pth')
    else:
        print('='*50, "Testing")
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        with torch.no_grad():
            res = model.decode(sample).cpu()
        torchvision.utils.save_image(
            res.view(n_samples*n_samples, channels_nbr, image_size, image_size).cpu(),
            f"./results/reconstruction_{epoch}.png",
            nrow=n_samples
        )

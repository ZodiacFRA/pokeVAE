#!/usr/bin/env python3
import sys
import time

import torch
import torchvision

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
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f"====> Epoch: {epoch} Average loss: {(train_loss / len(dataloader.dataset)):.4f}, Warmup Factor: {warmup_factor}")


if __name__ == '__main__':
    image_size = 64
    train_dataset = PokemonDataset(
        draw_samples=False,
        csv_file='./pokemons/pokemon.csv',
        root_dir='./pokemons/images',
        transform=torchvision.transforms.Compose([
            Rescale(image_size),
            ToTensor()
        ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    model = VAE(image_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    n_samples = 40
    # big_sample = get_sample(n, (-7, -7), (7, 7))
    rand_sample = torch.randn(n_samples*n_samples, LATENT_SPACE_SIZE).to(DEVICE)

    if len(sys.argv) == 1:
        print('='*50, "Training")
        for epoch in range(0, EPOCHS):
            warmup_factor = min(1, epoch / WARMUP_TIME)
            # Save preview
            if epoch % LOG_INTERVAL == 0:
                with torch.no_grad():
                    res = model.decode(rand_sample).cpu()
                torchvision.utils.save_image(
                    res.view(n_samples*n_samples, 1, image_size, image_size).cpu(),
                    f"./results/reconstruction_{epoch}.png",
                    nrow=n_samples // 2
                )

            train(epoch, warmup_factor, model, optimizer, train_dataloader)

        torch.save(model.state_dict(), f'./{time.time()}.pth')
    else:
        print('='*50, "Testing")
        model.load_state_dict(torch.load(sys.argv[1]))
        model.eval()
        with torch.no_grad():
            res = model.decode(sample).cpu()
        torchvision.utils.save_image(
            res.view(n_samples*n_samples, 1, image_size, image_size).cpu(),
            f"./results/reconstruction_{epoch}.png",
            nrow=n_samples // 2
        )

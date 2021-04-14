import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb

from utils import *


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.name_frame.iloc[idx, 0])
        image = io.imread(img_name + '.png')
        try:
            image = rgba2rgb(image)
        except ValueError:
            pass

        if self.transform:
            image = self.transform(image)
        return image


def show_name(image, name):
    """Show image with name"""
    plt.imshow(image)
    # plt.scatter(name[:, 0], name[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    face_dataset = PokemonDataset(
        csv_file='./pokemons/pokemon.csv',
        root_dir='./pokemons/images',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
        ]))

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['name'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(sample['name'])
        ax.axis('off')
        show_name(**sample)

        if i == 3:
            plt.show()
            break

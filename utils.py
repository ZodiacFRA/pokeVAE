import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray


from GLOBALS import *


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(sample, (new_h, new_w))
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample.astype(np.float32))


class SetChannels(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, output_size, channels_nbr, pad=True):
        self.channels_nbr = channels_nbr
        self.output_size = output_size
        self.pad = pad

    def __call__(self, sample):
        if self.channels_nbr == 1:
            sample = rgb2gray(sample)
            if self.pad:
                sample = sample[:self.output_size, :self.output_size, None]
            else:
                sample = sample[:self.output_size, :self.output_size]
        else:
            sample = sample[:self.output_size, :self.output_size, :self.channels_nbr]
        return sample


def get_sample(n, start, end):
    """ Create 2D representation by varying the value of each latent variable """
    props = []
    x1_l = np.arange(start[0], end[0], (end[0] - start[0]) / n)
    y1_l = np.arange(start[1], end[1], (end[1] - start[1]) / n)
    x2_l = np.arange(start[0], end[0], (end[0] - start[0]) / n)
    y2_l = np.arange(start[1], end[1], (end[1] - start[1]) / n)
    for x1 in x1_l:
        for y1 in y1_l:
            for x2 in x2_l:
                for y2 in y2_l:
                    props.append((x1, y1, x2, y2))
    return torch.FloatTensor(props).to(DEVICE)

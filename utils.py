import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

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


def get_sample(n, start, end):
    """ Create 2D representation by varying the value of each latent variable """
    props = []
    xl = np.arange(start[0], end[0], (end[0] - start[0]) / n)
    yl = np.arange(start[1], end[1], (end[1] - start[1]) / n)
    for x in xl:
        for y in yl:
            props.append((x, y))
    return torch.FloatTensor(props).to(DEVICE)

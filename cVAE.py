import torchvision

from GLOBALS import *


# class Flatten(torch.nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)


class UnFlatten(torch.nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class cVAE(torch.nn.Module):
    def __init__(self, image_size, image_channels):
        super(cVAE, self).__init__()
        self.pixels_nbr = image_size * image_size
        self.image_channels = image_channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.image_channels, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
        )
        # Latent
        self.mu = torch.nn.Linear(256*4, LATENT_SPACE_SIZE)
        self.var = torch.nn.Linear(256*4, LATENT_SPACE_SIZE)
        self.prepare_input_for_decoding = torch.nn.Linear(LATENT_SPACE_SIZE, 256*4)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256*4, 128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, self.image_channels, kernel_size=6, stride=2),
        )

    def encode(self, input):
        result = self.encoder(input)
        # Flatten the image for the linear layers
        result = result.view(result.size(0), -1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return self.mu(result), self.var(result)

    def decode(self, z):
        z = self.prepare_input_for_decoding(z)
        # Unflatten the latent values
        z = z.view(z.size(0), 1024, 1, 1)
        result = self.decoder(z)
        return torch.sigmoid(result)

    def forward(self, input):
        mu, var = self.encode(input)
        # Sample
        z = self.sample(mu, var)
        # Decode from sampled values
        res = self.decode(z)
        return res, mu, var

    def sample(self, mu, var):
        # Standard deviation
        std = torch.exp(0.5*var)
        # Sample from the distribution, std is only given as an indicator of the
        # shape needed, randn_like uses a mean of 0 and a variance of 1
        eps = torch.randn_like(std)
        # so multiply and add to take our values into account
        return mu + eps * std

    def loss_function(self, prediction, input, mu, var, warmup_factor):
        """ Reconstruction + KL divergence losses summed over all elements and batch """
        BCE = torch.nn.functional.binary_cross_entropy(prediction, input, reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        res = warmup_factor * KLD + BCE
        # print("===================================")
        # print('wu factor:', warmup_factor, 'KLD:', KLD, 'BCE:', BCE)
        # print('loss:', res)
        return res

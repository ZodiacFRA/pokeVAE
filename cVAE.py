import torchvision

from GLOBALS import *


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
            Flatten(),
        )
        # Latent
        self.mu = torch.nn.Linear(256*4, LATENT_SPACE_SIZE)
        self.var = torch.nn.Linear(256*4, LATENT_SPACE_SIZE)
        self.decoder_input = torch.nn.Linear(LATENT_SPACE_SIZE, 256*4)

        self.decoder = torch.nn.Sequential(
            UnFlatten(),
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
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return self.mu(result), self.var(result)

    def decode(self, z):
        result = self.decoder(z)
        return torch.sigmoid(result)

    def forward(self, input):
        print(input.shape)

        mu, var = self.encode(input)
        # Sample
        z = self.reparameterize(mu, var)
        z = self.decoder_input(z)
        # Decode from sampled values
        return self.decode(z), mu, var

    def reparameterize(self, mu, var):
        # Standard deviation
        std = torch.exp(0.5*var)
        # Sample from the distribution, std is only given as an indicator of the
        # shape needed, randn_like uses a mean of 0 and a variance of 1
        eps = torch.randn_like(std)
        # so multiply and add to take our values into account
        return mu + eps * std

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, prediction, x, mu, var, warmup_factor):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        BCE = torch.nn.functional.binary_cross_entropy(prediction, x.view(-1, self.pixels_nbr), reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        res = warmup_factor * KLD + BCE
        # print("===================================")
        # print('wu factor:', warmup_factor, 'KLD:', KLD, 'BCE:', BCE)
        # print('loss:', res)
        return res


def test(epoch, model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.to(DEVICE)
            recon_batch, mu, var = model(data)
            test_loss += loss_function(recon_batch, data, mu, var, 1).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
            #     torchvision.utils.save_image(comparison.cpu(), './results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(dataloader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

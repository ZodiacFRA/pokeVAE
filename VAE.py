import torchvision

from GLOBALS import *


class VAE(torch.nn.Module):
    def __init__(self, image_size):
        super(VAE, self).__init__()
        self.pixels_nbr = image_size * image_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.pixels_nbr, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        # Latent
        self.mu = torch.nn.Linear(256, LATENT_SPACE_SIZE)
        self.var = torch.nn.Linear(256, LATENT_SPACE_SIZE)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(LATENT_SPACE_SIZE, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.pixels_nbr),
            torch.nn.ReLU(),
        )

    def encode(self, input):
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        log_var = self.var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder(z)
        return torch.sigmoid(result)

    def forward(self, x):
        mu, var = self.encode(x.view(-1, self.pixels_nbr))
        z = self.reparameterize(mu, var)
        return self.decode(z), mu, var

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
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

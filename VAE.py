import torchvision

from GLOBALS import *


class VAE(torch.nn.Module):
    def __init__(self, image_size):
        super(VAE, self).__init__()
        self.pixels_nbr = image_size * image_size
        # Encoder
        self.fe1 = torch.nn.Linear(self.pixels_nbr, 2048)
        self.fe2 = torch.nn.Linear(2048, 1024)
        self.fe3 = torch.nn.Linear(1024, 512)
        self.fe4 = torch.nn.Linear(512, 256)
        # Latent
        self.mu = torch.nn.Linear(256, LATENT_SPACE_SIZE)
        self.logvar = torch.nn.Linear(256, LATENT_SPACE_SIZE)
        # Decoder
        self.fd1 = torch.nn.Linear(LATENT_SPACE_SIZE, 256)
        self.fd2 = torch.nn.Linear(256, 512)
        self.fd3 = torch.nn.Linear(512, 1024)
        self.fd4 = torch.nn.Linear(1024, 2048)
        self.fd5 = torch.nn.Linear(2048, self.pixels_nbr)

    def encode(self, x):
        h1 = torch.nn.functional.relu(self.fe1(x))
        h1 = torch.nn.functional.relu(self.fe2(h1))
        h1 = torch.nn.functional.relu(self.fe3(h1))
        h1 = torch.nn.functional.relu(self.fe4(h1))
        return self.mu(h1), self.logvar(h1)

    def decode(self, z):
        h1 = torch.nn.functional.relu(self.fd1(z))
        h1 = torch.nn.functional.relu(self.fd2(h1))
        h1 = torch.nn.functional.relu(self.fd3(h1))
        h1 = torch.nn.functional.relu(self.fd4(h1))
        return torch.sigmoid(self.fd5(h1))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.pixels_nbr))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, prediction, x, mu, logvar, warmup_factor):
        BCE = torch.nn.functional.binary_cross_entropy(prediction, x.view(-1, self.pixels_nbr), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        res = warmup_factor * KLD + BCE
        res = min(res, 50_000)
        return res


def test(epoch, model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, 1).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
            #     torchvision.utils.save_image(comparison.cpu(), './results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(dataloader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")

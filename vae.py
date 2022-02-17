import torch
from torch import nn
from torch.functional import F

from torch.distributions import *

from utilities import softplus

######################################################
# Encoder
###########################

class Encoder(nn.Module):
    """
    Decription

    Parameters:
    -----------


    Returns:
    --------


    """
    def __init__(self, inputs_dim, latent_dim):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(inputs_dim, 2 * latent_dim),
        )

    def forward(self, inputs):
        return self.mlp(inputs).chunk(2, dim=-1)


######################################################
# Decoder
###########################

class Decoder(nn.Module):
    """
    Decription

    Parameters:
    -----------


    Returns:
    --------


    """
    def __init__(self, latent_dim, outputs_dim):
        super(Decoder, self).__init__()

        self.inputs_dim = latent_dim
        self.outputs_dim = outputs_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.inputs_dim, self.outputs_dim),
        )

    def forward(self, z):

        """
        Decription

        Parameters:
        -----------

        Returns:
        --------


        """

        a_hat = torch.sigmoid(self.mlp(z.float())).T

        return a_hat.T


######################################################
# VAE
###########################

class VAE(nn.Module):
    """
     Decription

     Parameters:
     -----------

     Returns:
     --------


     """
    def __init__(self, a_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(a_dim, latent_dim)
        self.decoder = Decoder(latent_dim, a_dim)


    def forward(self, a):

        z_mean_q, z_logvar_q = self.encoder(a.float())
        z_sample = Normal(z_mean_q, z_logvar_q).rsample()

        a_hat = self.decoder(z_sample)

        return z_mean_q, z_logvar_q, a_hat, z_sample


    def loss(self, a, a_hat, z_mean_q, z_logvar_q):

        mu_q, sigma_q = z_mean_q.view(-1), torch.sqrt(softplus(z_logvar_q.view(-1)))
        mu_p, sigma_p = torch.zeros(z_mean_q.size(0)), torch.ones(z_mean_q.size(0))

        q = Normal(mu_q, sigma_q)
        p = Normal(mu_p, sigma_p)

        kl = torch.distributions.kl.kl_divergence(q, p).mean()

        recon_loss = F.binary_cross_entropy(a_hat.reshape(-1), a.reshape(-1))

        weight = 1000
        return  recon_loss + weight * kl, recon_loss, kl

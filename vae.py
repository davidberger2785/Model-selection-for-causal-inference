import numpy as np

import torch
from torch import nn
from torch.functional import F

from torch.distributions import *

from utilities import select, softplus


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
            nn.Linear(inputs_dim, 2 * latent_dim, bias=False),
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
    def __init__(self, latent_dim, a_dim, order=0, outputs_dim=1):
        super(Decoder, self).__init__()

        self.order = order
        self.a_dim = a_dim
        self.inputs_dim = latent_dim + order
        self.outputs_dim = outputs_dim


        self.mlp = nn.Sequential(
            nn.Linear(self.inputs_dim, self.outputs_dim),
        )



    def forward(self, z, y=None):

        """
        Decription

        Parameters:
        -----------

        Returns:
        --------


        """
        m = nn.LogSoftmax()

        a_hat = torch.empty((self.a_dim, z.size(0)))

        if self.order != 0:
            for k in np.arange(self.a_dim):
                inputs = torch.cat((z, select(y, k, self.order)), -1)
                a_hat[k] = torch.sigmoid(self.mlp(inputs.float()).view(-1))
        else:
            a_hat = torch.sigmoid(self.mlp(z.float())).T

        return a_hat.T


######################################################
# CVAE
###########################

class CVAE(nn.Module):
    """
     Decription

     Parameters:
     -----------
     dsfa

     Returns:
     --------


     """
    def __init__(self, a_dim, y_dim, latent_dim, order, kl_coef, kl_begin, kl_end):
        super(CVAE, self).__init__()

        self.encode_posterior = Encoder(a_dim+y_dim, latent_dim)
        self.encode_prior = Encoder(y_dim, latent_dim)
        self.decode = Decoder(latent_dim, a_dim, order)

        self.kl_coef = kl_coef
        self.kl_begin = kl_begin
        self.kl_end = kl_end

    def forward(self, a, y):

        # According to the ELBO in Section 6
        z_mean_q, z_logvar_q = self.encode_posterior(torch.cat((a, y), -1).float())
        z_mean_p, z_logvar_p = self.encode_prior(y.float())

        # parameterization trick
        z_sample = Normal(z_mean_q, torch.sqrt(softplus(z_logvar_q))).rsample()

        a_hat = self.decode(z_sample, y)

        return z_mean_q, z_logvar_q, z_mean_p, z_logvar_p, a_hat, z_sample

    def loss(self, a, a_hat, z_mean_q, z_logvar_q, z_mean_p, z_logvar_p):

        mu_q, sigma_q = z_mean_q.view(-1), torch.sqrt(softplus(z_logvar_q.view(-1)))
        mu_p, sigma_p = z_mean_p.view(-1), torch.sqrt(softplus(z_logvar_p.view(-1)))

        q = Normal(mu_q, sigma_q)
        p = Normal(mu_p, sigma_p)

        kl = torch.distributions.kl.kl_divergence(q, p).mean()

        recon_loss = F.binary_cross_entropy(a_hat.reshape(-1), a.reshape(-1), reduction='mean')

        return recon_loss + self.kl_coef * kl, recon_loss, kl


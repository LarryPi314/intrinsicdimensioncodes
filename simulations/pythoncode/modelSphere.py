
import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, w, q):
        """
        Initialize generator

        :param w: number of channels on the finest level
        :param q: latent space dimension
        """
        super(Generator, self).__init__()
        self.w = w
        self.fc = nn.Linear(q, w)
        #self.bn1 = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(w, 3)

    def forward(self, z):
        """
        :param z: latent space sample
        :return: g(z)
        """
        gz = F.relu(self.fc(z))
        #gz = self.bn1(gz)
        gz = torch.tanh(self.fc2(gz))  # Adjusted to use tanh activation for bounded output
        return gz


class Encoder(nn.Module):
    def __init__(self, w, q):
        """
        Initialize the encoder for the VAE

        :param w: number of channels on finest level
        :param q: latent space dimension
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1, w)
        self.fc_mu = nn.Linear(w, q)
        self.fc_logvar = nn.Linear(w, q)

    def forward(self, x):
        """
        :param x: tensor with shape (batch_size, 3)
        :return: mu, logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar






import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import torch.nn.functional as F
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, q):
        """
        Initialize generator

        :param w: number of channels on the finest level
        :param q: latent space dimension
        """
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(q, 128)
        self.linear2_mu = nn.Linear(128, 3)
        self.fixed_scale = nn.Parameter(torch.tensor(0.)) # logvar
    
        self.bn1 = nn.BatchNorm1d(128)


    def forward(self, z):
        """
        :param z: latent space sample
        :return: g(z)
        """
        gz = self.linear1(z)
        gz = self.bn1(gz)
        gz = F.relu(gz)
        gz = self.linear2_mu(gz)
        gz = torch.sigmoid(gz)

        return gz
        # h = F.relu(self.linear1(z))
        # mu = torch.sigmoid(self.linear2_mu(h))
        # logvar = self.fixed_scale.expand_as(mu)
        
        # sigma = torch.exp(logvar / 2)
        # normal_dist = torch.distributions.Normal(mu, sigma)
        # xhat = normal_dist.rsample()
        # xhat = torch.sigmoid(xhat)
        # return xhat
    

q = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = Generator(q)
checkpoint_path = path.join('results', 'VAEmnist2-q-2-batch_size-64-g.pt')
generator.load_state_dict(torch.load(checkpoint_path))
generator.to(device)
generator.eval()

# Visualize results

# z1 = -torch.ones(1,2,device=device)
# z2 = torch.ones(1,2,device=device)
# with torch.no_grad():
#     lam = torch.linspace(0,1,50,device=device).reshape(-1,1)
#     zinter = z1*(1-lam) + z2*lam
#     gz_vae = generator(zinter)

def sample_sphere(n):
    z = torch.randn(n, 3)
    z = z / z.norm(dim=1).reshape(-1, 1)
    return z

latent_vector_1 = torch.randn(1000, q).to(device)
# latent_vector_2 = torch.linspace(-10, 10, 100).to(device).reshape(-1, q)

with torch.no_grad():
    gz_vae_1 = generator(latent_vector_1)
    # gz_vae_2 = generator(latent_vector_2)

sphere = sample_sphere(1000)

fig = plt.figure(facecolor='lightskyblue')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gz_vae_1[:, 0], gz_vae_1[:, 1], gz_vae_1[:, 2], color = "blue")
ax.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], color = "green")
# ax.scatter(gz_vae_2[:, 0], gz_vae_2[:, 1], gz_vae_2[:, 2], color = "red")

plt.show()
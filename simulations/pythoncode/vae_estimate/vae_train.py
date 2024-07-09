from vae import *
import numpy as np
import matplotlib.pyplot as plt
from os import path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare training data
# Custom dataset class
from torch.utils.data import Dataset
class UnitSphereDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random point on the unit sphere
        point = np.random.randn(3)
        point /= np.linalg.norm(point)  # Normalize to unit length
        return torch.tensor(point, dtype=torch.float32)
    
batch_size = 64

# Load the dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_dataset = UnitSphereDataset(50000)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

test_dataset = UnitSphereDataset(50000)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

q = 2

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

class Encoder(nn.Module):
    def __init__(self,q):
        """
        Initialize the encoder for the VAE

        :param w: number of channels on finest level
        :param q: latent space dimension
        """
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, q) # outputs mu
        self.linear3 = nn.Linear(128, q) # outputs logvar
        
        self.N = torch.distributions.Normal(0, 1)

        
    def forward(self, x):
        """
        :param x: MNIST image
        :return: mu,logvar that parameterize e(z|x) = N(mu, diag(exp(logvar)))
        """
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        logvar = self.linear3(x)

        return mu, logvar
    
g = Generator(q)
e = Encoder(q)

vae = VAE(e,g).to(device)

# Train the VAE
retrain = False
out_file = "./results/VAEmnist2-q-2-batch_size-%d" % (batch_size)

num_epochs = 40
optimizer = torch.optim.Adam(params=vae.parameters(), lr=1e-3, weight_decay=1e-5)

his = np.zeros((num_epochs,6))
print((3*"--" + "device=%s, q=%d, batch_size=%d, num_epochs=%d" + 3*"--") % (device, q, batch_size, num_epochs))

if out_file is not None:
    import os
    out_dir, fname = os.path.split(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3*"--" + "out_file: %s" + 3*"--") % (out_file))

print((7*"%7s    ") % ("epoch","Jtrain","pzxtrain","ezxtrain","Jval","pzxval","ezxval"))

for epoch in range(num_epochs):
    vae.train()

    train_loss = 0.0
    train_pzx = 0.0
    train_ezx = 0.0
    num_ex = 0
    for image_batch in train_dataloader:
        image_batch = image_batch.to(device)

        # take a step
        loss, pzx, ezx,gz,mu = vae.ELBO(image_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update history
        train_loss += loss.item()*image_batch.shape[0]
        train_pzx += pzx*image_batch.shape[0]
        train_ezx += ezx*image_batch.shape[0]
        num_ex += image_batch.shape[0]

    train_loss /= num_ex
    train_pzx /= num_ex
    train_ezx /= num_ex

    # evaluate validation points
    vae.eval()
    val_loss = 0.0
    val_pzx = 0.0
    val_ezx = 0.0
    num_ex = 0
    for image_batch in test_dataloader:
        with torch.no_grad():
            image_batch = image_batch.to(device)
            # vae reconstruction
            loss, pzx, ezx, gz, mu = vae.ELBO(image_batch)
            val_loss += loss.item() * image_batch.shape[0]
            val_pzx += pzx * image_batch.shape[0]
            val_ezx += ezx * image_batch.shape[0]
            num_ex += image_batch.shape[0]

    val_loss /= num_ex
    val_pzx/= num_ex
    val_ezx/= num_ex

    print(("%06d   " + 6*"%1.4e  ") %
            (epoch + 1, train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx))

    his[epoch,:] = [train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx]

if out_file is not None:
    torch.save(vae.g.state_dict(), ("%s-g.pt") % (out_file))
    torch.save(vae.state_dict(), ("%s.pt") % (out_file))
    from scipy.io import savemat
    savemat(("%s.mat") % (out_file), {"his":his})


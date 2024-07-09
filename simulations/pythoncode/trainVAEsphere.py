
from vae import *
from torch import distributions
from torch import nn
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils import sample_uniform_sphere

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('VAE')
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--q", type=int, default=2, help="latent space dimension")
parser.add_argument("--width_enc", type=int, default=4, help="width of encoder")
parser.add_argument("--width_dec", type=int, default=4, help="width of decoder")
parser.add_argument("--num_epochs", type=int, default=2, help="number of epochs")
parser.add_argument("--out_file", type=str, default=None, help="base filename saving trained model (extension .pt), history (extension .mat), and intermediate plots (extension .png")
args = parser.parse_args()

from modelSphere import Encoder, Generator
g = Generator(args.width_dec, args.q)
e = Encoder(args.width_enc, args.q)

vae = VAE(e, g).to(device)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=1e-4, weight_decay=1e-5)

his = np.zeros((args.num_epochs, 6))

print((3 * "--" + "device=%s, q=%d, batch_size=%d, num_epochs=%d, w_enc=%d, w_dec=%d" + 3 * "--") % (device, args.q, args.batch_size, args.num_epochs, args.width_enc, args.width_dec))

if args.out_file is not None:
    import os
    out_dir, fname = os.path.split(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print((3 * "--" + "out_file: %s" + 3 * "--") % (args.out_file))

print((7 * "%7s    ") % ("epoch", "Jtrain", "pzxtrain", "ezxtrain", "Jval", "pzxval", "ezxval"))

# Generate training data
train_data = sample_uniform_sphere(10000)  # Increased samples
test_data = sample_uniform_sphere(1000)   # Increased samples

train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(2))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32).unsqueeze(2))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

for epoch in range(args.num_epochs):
    vae.train()

    train_loss = 0.0
    train_pzx = 0.0
    train_ezx = 0.0
    num_ex = 0
    for batch in train_dataloader:
        data_batch = batch[0].to(device)

        # take a step
        loss, pzx, ezx, gz, mu = vae.ELBO(data_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update history
        train_loss += loss.item() * data_batch.shape[0]
        train_pzx += pzx * data_batch.shape[0]
        train_ezx += ezx * data_batch.shape[0]
        num_ex += data_batch.shape[0]

    train_loss /= num_ex
    train_pzx /= num_ex
    train_ezx /= num_ex

    # evaluate validation points
    vae.eval()
    val_loss = 0.0
    val_pzx = 0.0
    val_ezx = 0.0
    num_ex = 0
    for batch in test_dataloader:
        with torch.no_grad():
            data_batch = batch[0].to(device)
            # vae reconstruction
            loss, pzx, ezx, gz, mu = vae.ELBO(data_batch)
            val_loss += loss.item() * data_batch.shape[0]
            val_pzx += pzx * data_batch.shape[0]
            val_ezx += ezx * data_batch.shape[0]
            num_ex += data_batch.shape[0]

    val_loss /= num_ex
    val_pzx /= num_ex
    val_ezx /= num_ex

    print(("%06d   " + 6 * "%1.4e  ") % (epoch + 1, train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx))

    his[epoch, :] = [train_loss, train_pzx, train_ezx, val_loss, val_pzx, val_ezx]

if args.out_file is not None:
    torch.save(vae.g.state_dict(), ("%s-g.pt") % (args.out_file))
    torch.save(vae.state_dict(), ("%s.pt") % (args.out_file))
    from scipy.io import savemat
    savemat(("%s.mat") % (args.out_file), {"his": his})

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming test_samples and gz_vae are already computed as in your previous code

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Original Test Samples on Sphere (3D)
test_samples = next(iter(test_dataloader))[0].numpy()
if test_samples.shape[1] == 1:
    test_samples = test_samples.squeeze(1)

ax.scatter(test_samples[:, 0], test_samples[:, 1], test_samples[:, 2], c='blue', marker='o', label='Original Test Samples')

# Generate new samples and plot (3D)
z = torch.randn((64, args.q), device=device)
gz_vae = vae.g(z).cpu().detach().numpy()

ax.scatter(gz_vae[:, 0], gz_vae[:, 1], gz_vae[:, 2], c='red', marker='^', label='Generated Samples')

ax.set_title('3D Plot of Original and Generated Samples')
ax.legend()

plt.tight_layout()
plt.show()

# Interpolated Samples (3D)
'''
data_samples = next(iter(test_dataloader))[0][:2].to(device)
mu, _ = vae.e(data_samples)
z1 = mu[0].unsqueeze(0)  # First data point's mean vector
z2 = mu[1].unsqueeze(0)  # Second data point's mean vector

with torch.no_grad():
    lam = torch.linspace(0, 1, 20, device=device).reshape(-1, 1)
    zinter = z1 * (1 - lam) + z2 * lam  # Ensure z1 and z2 have the same shape
    gz_vae = vae.g(zinter).cpu().numpy()

ax.scatter(gz_vae[:, 0], gz_vae[:, 1], gz_vae[:, 2], c='green', marker='x', label='Interpolated Samples')

ax.set_title('3D Plot of Original, Generated, and Interpolated Samples')
ax.legend()

plt.tight_layout()
plt.show()
'''




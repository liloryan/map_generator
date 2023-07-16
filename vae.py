#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.autograd import Variable
#%%
data_dir = Path('data')
models_dir = Path('models')
models_dir.mkdir(parents = True, exist_ok = True)
image_size = 256

batch_size = 32
epochs = 1000

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
full_dataset = torchvision.datasets.ImageFolder(root = data_dir)
train_size = int(len(full_dataset)*0.8)
test_size = int(len(full_dataset) - train_size)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_transform = transforms.Compose(
    [
        transforms.Resize(size = image_size),
        transforms.RandomCrop(size = (image_size,image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(size = image_size),
        transforms.CenterCrop(size = (image_size,image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]
)

train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
#%%
def imshow(img):
    """function to show image"""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

images,labels = next(iter(train_loader))
imshow(torchvision.utils.make_grid(images))


# %%
# modified from: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU()
            )
        )
        self.linear1 = nn.Linear(32768, 128)
        self.linear_mu = nn.Linear(128, latent_dims)
        self.linear_sigma = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        else:
            self.N.loc = self.N.loc.cpu() 
            self.N.scale = self.N.scale.cpu()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.leaky_relu(self.linear1(x))
        mu =  self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z      

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32768),
            nn.LeakyReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 8, 8))

        self.unconvs = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, out_channels= 1, kernel_size= 3, padding= 1),
                nn.Tanh()
            )
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.unconvs(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
#%%

def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        ms_error = ((x - x_hat)**2).sum()
        loss = ms_error/750 + vae.encoder.kl
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
       #print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()


    return train_loss / len(dataloader.dataset)
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum()/750 + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)
def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

#%%
latent_dims = 64
model_name = f'vae_{latent_dims}'
vae = VariationalAutoencoder(latent_dims)
optim = torch.optim.AdamW(vae.parameters(), lr=1e-5)
vae.to(device)
#%%
train_losses = []
val_losses = []
for epoch in range(epochs):
    train_loss = train_epoch(vae,device, train_loader,optim)
    val_loss = test_epoch(vae,device, test_loader)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, epochs,train_loss,val_loss))
    plot_ae_outputs(vae.encoder,vae.decoder,n=10)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    torch.save(vae.state_dict(), models_dir/f'{model_name}.{epoch}.pth')
    
#%%
plt.plot(train_losses)
plt.plot(val_losses)
#%%
vae.eval()

with torch.no_grad():

    # sample latent vectors from the normal distribution
    latent = torch.randn(128, latent_dims, device=device)

    # reconstruct images from the latent vectors
    img_recon = vae.decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(20, 8.5))
    imshow(torchvision.utils.make_grid(img_recon.data[:100],10,5))
    plt.show()


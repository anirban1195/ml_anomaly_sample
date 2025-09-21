#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 12:47:29 2025

@author: dutta26
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
from tqdm import tqdm

# Configuration
########################################################
DATA_DIR = "/scratch/bell/dutta26/mvtec/bottle/train/good/"      # Directory with 'good' training images
IMAGE_SIZE = 256             # Image size for the GAN
BATCH_SIZE = 32
EPOCHS = 1000                 # GANs require more epochs and low LR to converge
LATENT_DIM = 100             # Size of the noise vector z
LEARNING_RATE = 1e-4
BETA1 = 0.5                  # Adam optimizer beta1
BETA2 = 0.99                # Adam optimizer beta2
N_CRITIC = 5                 # Train critic 5 times per generator update
LAMBDA_GP = 10               # Gradient penalty coefficient
SAMPLE_INTERVAL = 280        # Save sample images every N batches (For Visual inspection of training)
MODELS_DIR = "gan_models"    # Directory to save trained models
SAMPLES_DIR = "gan_samples"  # Directory to save generated image samples (Visual inspection of training)
#########################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Data Augmentation protocol
# Do flips up-down, left-right and rotation/shift
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(degrees=360, fill=1), #Double check here. FILL 1?
    transforms.RandomAffine(
        degrees=360,
        translate=(0.02, 0.02),
        fill=255 # White background fill
    ),
    transforms.ToTensor(), #Converts to [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


#Custom bottle dataset
class BottleDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Model Architecture
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



#Simple generator and discriminator
#BatchNorm seems to work better here. Likely because we want a 
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 16 x16
            nn.ConvTranspose2d(256, 128, 4, 2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
             # 64 x 64 x 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32x128x128
            nn.ConvTranspose2d(32, 3,4, 2, 1),
            nn.Tanh()
            # final state size. 3 x 256 x 256
        )
    #Perform forward pass
    def forward(self, input):
        return self.main(input)

# A VERY simple discriminator
#Instance norm works better here because I think each sample must be 
#normalized and normalization from other samples should not affect the decision
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # input is 3 x 128 x 128
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            #64x64x 64
            nn.Conv2d(64, 128, 4, 2,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            #128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # 256x16x16
            nn.Conv2d(256,512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            #512x8x8
            nn.Conv2d(512, 1024, 4,2, 1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            #1024 x4 x4
            nn.Conv2d(1024, 1, 4, 1, 0)
            # final state size. 1x1x1 (a single scalar)
        )
    #Def do the forward pass
    def forward(self, input):
        return self.main(input)

# Gradient penalty script
# this is to stabilize the training
# Not sure Iif I fully understand the math/reason here
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=DEVICE)
    #Make images that are alpah between fake and real
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = discriminator(interpolates)  #Use discrimiator of these mixed images

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    #Flatten and calculate the L2 norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training Script 
if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # Setup dataset and dataloader
    dataset = BottleDataset(directory=DATA_DIR, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize models and apply weights
    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers. Standard method for GANs to specify beta1 and beta2
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Fixed noise for consistent sampling
    # Helpful to vizualize the learning process
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    print("Starting Training")
    batches_done = 0
    for epoch in range(EPOCHS):
        for i, real_images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_images = real_images.to(DEVICE)

            # Train Critic /Discriminator
            optimizer_d.zero_grad()

            # Generate a batch of fake images. Used for grad penalty
            #and dicriminator training
            noise = torch.randn(real_images.size(0), LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise)

            #Scores for real and fake images
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images.detach()) #DONT use fake images for backprop (flow to generator)

            #Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)

            # Critic loss
            # push real_validits --> inf fake_validity --> -inf + gradient penalty
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

            d_loss.backward()
            optimizer_d.step()

            # Train generator only once every N_CRITIC passes
            if i % N_CRITIC == 0:
                optimizer_g.zero_grad()

                # Generate a new batch of fake images
                gen_imgs =generator(noise)

                # Generator loss (wants discriminator to score its fakes as real)
                # since gen_images must be =  real_images ,discriminator(gen_imgs) --> inf 
                g_loss = -torch.mean(discriminator(gen_imgs))

                g_loss.backward()
                optimizer_g.step()

                # Mkase some sample to vizeulaize the learning process 
                if batches_done% SAMPLE_INTERVAL == 0:
                    print(
                        f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                    )
                    with torch.no_grad():
                        gen_samples = generator(fixed_noise).detach().cpu()
                    save_image(gen_samples, f"{SAMPLES_DIR}/{batches_done}.png", nrow=8, normalize=True)

                batches_done += N_CRITIC

    print("Finished Training.")

    # Save final models
    torch.save(generator.state_dict(), os.path.join(MODELS_DIR, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(MODELS_DIR, "discriminator.pth"))
    print(f"Models saved in '{MODELS_DIR}' directory.")

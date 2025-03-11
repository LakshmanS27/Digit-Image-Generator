import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt

# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),                  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))    # Normalize pixel values to [-1, 1]
])

# Load the dataset
batch_size = 128  # Number of images per batch
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check dataset shape
dataiter = iter(dataloader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}")  # Expected: (128, 1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(100, 256),       # Input: Random noise (100) → 256 neurons
            nn.ReLU(),
            nn.Linear(256, 512),       # 256 → 512
            nn.ReLU(),
            nn.Linear(512, 1024),      # 512 → 1024
            nn.ReLU(),
            nn.Linear(1024, 28*28),    # 1024 → 784 (Flattened MNIST image)
            nn.Tanh()                  # Normalize output to [-1, 1] (matches dataset)
        )

    def forward(self, z):
        output = self.model(z)
        output = output.view(-1, 1, 28, 28)  # Reshape to image format (1, 28, 28)
        return output

# Create an instance of the generator
generator = Generator()
print(generator)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),  # Input: Flattened image (784 pixels) → 1024 neurons
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),    # 1024 → 512
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),     # 512 → 256
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),       # 256 → 1 (single probability)
            nn.Sigmoid()             # Output is between [0, 1] (real or fake)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)  # Flatten (1, 28, 28) → (784)
        output = self.model(img_flat)
        return output

# Create an instance of the discriminator
discriminator = Discriminator()
print(discriminator)

# Loss function (Binary Cross Entropy)
loss_function = nn.BCELoss()  

# Optimizers for Generator and Discriminator
lr = 0.0002  # Learning rate

generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training parameters
num_epochs = 50  # You can adjust based on available time
latent_dim = 100  # Size of random noise input to the generator

# Start Training
for epoch in range(num_epochs):
    start_time = time.time()
    
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.shape[0]
        
        # REAL IMAGES (Label = 1)
        real_labels = torch.ones(batch_size, 1)  
        
        # FAKE IMAGES (Label = 0)
        fake_labels = torch.zeros(batch_size, 1)
        
        ###### Train Discriminator ######
        discriminator_optimizer.zero_grad()

        # Compute loss for real images
        real_images = real_images.view(batch_size, -1)  # Flatten images
        real_preds = discriminator(real_images)  # Discriminator output
        real_loss = loss_function(real_preds, real_labels)

        # Generate fake images
        noise = torch.randn(batch_size, latent_dim)  # Random noise
        fake_images = generator(noise)

        # Compute loss for fake images
        fake_preds = discriminator(fake_images.detach())  # Detach to avoid training generator
        fake_loss = loss_function(fake_preds, fake_labels)

        # Total Discriminator Loss & Backpropagation
        d_loss = real_loss + fake_loss
        d_loss.backward()
        discriminator_optimizer.step()

        ###### Train Generator ######
        generator_optimizer.zero_grad()

        # Generate new fake images
        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        
        # Fool the discriminator (we want fake images to be classified as real)
        fake_preds = discriminator(fake_images)
        g_loss = loss_function(fake_preds, real_labels)  # Generator wants fake_preds to be 1

        # Backpropagation for Generator
        g_loss.backward()
        generator_optimizer.step()
        
    # Print progress
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_time:.2f}s")

# Save the trained generator
torch.save(generator.state_dict(), "generator.pth")
print("Generator model saved as 'generator.pth'")
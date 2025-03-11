import torch
import matplotlib.pyplot as plt
import numpy as np
from train_gan import Generator  # Import the Generator class

# Load the trained generator
generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()  # Set to evaluation mode

# Function to generate and display images
def generate_and_show_images(generator, num_images=16):
    noise = torch.randn(num_images, 100)  # Generate random noise
    fake_images = generator(noise).detach().cpu().numpy()  # Generate images
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i][0], cmap='gray')  # Display image
        ax.axis('off')
    
    plt.show()

# Generate and display fake images
generate_and_show_images(generator)

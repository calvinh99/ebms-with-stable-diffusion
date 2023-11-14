import os
import numpy as np
import torch
import torchvision.utils as vutils
from diffusers import AutoencoderKL
import torchvision.transforms as transforms
import importlib
import yaml
import argparse
import random
import matplotlib.pyplot as plt
import csv
from PIL import Image

def resize_images(images, size=(32, 32)):
    resized_images = []
    transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Unnormalize
        transforms.ToPILImage(),
        transforms.Resize(size, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Re-normalize
    ])

    for img in images:
        img = img.cpu()
        img_resized = transform(img)
        resized_images.append(img_resized)
    
    return torch.stack(resized_images)

# get config path
parser = argparse.ArgumentParser(description='Experiment Configuration')
parser.add_argument('config_path', type=str, help='Path to the config.yaml file')
# parser.add_argument('checkpoint_iter', type=int, help='Which version of the checkpoint to use')
args = parser.parse_args()

# Extract experiment name from config file name
experiment_name = os.path.splitext(os.path.basename(args.config_path))[0]
print(f"EXPERIMENT: {experiment_name}")

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize experiment directories
log_dir = os.path.join(config['log_root'], experiment_name)
imgs_dir = os.path.join(log_dir, 'imgs')
weights_dir = os.path.join(log_dir, 'weights')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Initialize a CSV file to save the gradients and noise
csv_file_path = os.path.join(log_dir, 'gradient_noise_data.csv')
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['iteration', 'gradient_norm', 'noise_norm'])  # Step 2: Write header

# Init hyperparams
n_iters = config['n_iters']
m = config['m']
K = config['K']
step_size = float(config['step_size'])
lr = float(config['lr'])
sigma = float(config['sigma'])
mcmc_noise = float(config['mcmc_noise'])
dtype = getattr(torch, config.get('dtype', 'float32'))
seed = config['seed']
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get vae
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-2", subfolder="vae")
vae.to(device)

# image to latent code (vae produces the distribution, we sample from this)
def encode_images(vae, images):
    latents = []
    for image in images:
        with torch.no_grad():
            latent = vae.encode(image.unsqueeze(0)).latent_dist.sample()
            latent = latent.to(dtype)
        latents.append(latent)
    return torch.cat(latents)

# get dataset from config
dataset_module = importlib.import_module(config['dataset_module'])
dataset = getattr(dataset_module, 'get_dataset')()

# get latent shape
sample_img = dataset[0][0].unsqueeze(0) # we don't want labels, just the image
sample_img = sample_img.to(device)
sample_latent = encode_images(vae, sample_img)
print(f"LATENT SHAPE: {sample_latent.shape}")
_, n_channels, latent_width, latent_height = sample_latent.shape

# get ebm architecture from config
model_module = importlib.import_module(config['ebm_module'])
create_model_from_config_fn = getattr(model_module, 'create_model_from_config')
ebm = create_model_from_config_fn(config)
ebm = ebm.to(device).to(dtype)
print(f"Number of parameters: {sum(p.numel() for p in ebm.parameters() if p.requires_grad)}")
print(f"MODEL ARCHITECTURE\n {ebm}")

# Update Adam optimizer if dtype is float16
optimizer = torch.optim.Adam(ebm.parameters(), lr=lr, betas=[0.9, 0.999])
if dtype == torch.float16:
    optimizer = torch.optim.Adam(ebm.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-4)

# ebm functions
def sample_from_data_distribution(dataset, batch_size, sigma): 
    indices = random.sample(range(len(dataset)), batch_size)
    batch_images = torch.stack([dataset[i][0] for i in indices])
    batch_images = batch_images.to(device)
    latents = encode_images(vae, batch_images).to(dtype)
    return latents + sigma * torch.randn_like(latents)

def sample_from_model(ebm, n_steps, step_size, batch_size, 
                      n_channels, latent_width, latent_height, dtype):
    samples = torch.randn(batch_size, n_channels, latent_width, latent_height, 
                          device=device, dtype=dtype, requires_grad=True)
    for _ in range(n_steps):
        ebm_sum = ebm(samples).sum()
        gradient = torch.autograd.grad(ebm_sum, [samples], retain_graph=True)[0]
        # noise_term = np.sqrt(step_size) * torch.randn_like(samples)
        noise_term = mcmc_noise * np.sqrt(step_size) * torch.randn_like(samples)
        samples.data += step_size/2 * gradient + noise_term
    return samples.detach(), torch.norm(gradient).item(), torch.norm(noise_term).item()


# training loop
for i in range(0, n_iters):
    real_samples = sample_from_data_distribution(dataset, m, sigma)
    model_samples, gradient_norm, noise_norm = sample_from_model(
        ebm, K, step_size, m, n_channels, latent_width, latent_height, dtype)
    
    # calculate differnce in energy values
    loss = ebm(real_samples).mean() - ebm(model_samples).mean()
    optimizer.zero_grad()
    (-loss).backward()
    optimizer.step()

    # Writing the norms to the CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([i, gradient_norm, noise_norm])

    if i % 100 == 0:
        n_samples = 64
        generated_samples, _, _ = sample_from_model(ebm, K, step_size, n_samples, n_channels, latent_width, latent_height, dtype)
        generated_samples = generated_samples.to(dtype=torch.float32)
        decoded_samples = []
        for j in range(n_samples):
            with torch.no_grad():
                decoded_sample = vae.decode(generated_samples[j].unsqueeze(0)).sample
            decoded_samples.append(decoded_sample)
        decoded_samples = torch.cat(decoded_samples, dim=0)

        # Get 16 real images from dataset
        real_indices = random.sample(range(len(dataset)), n_samples)
        real_images = torch.stack([dataset[i][0] for i in real_indices]).to(device)

        # Resize real and generated images
        real_images_resized = resize_images(real_images)
        decoded_samples_resized = resize_images(decoded_samples)

        # Concatenate resized real and generated images
        all_images_resized = torch.cat((real_images_resized, decoded_samples_resized), dim=0)

        # Save concatenated resized samples
        save_path_resized = os.path.join(imgs_dir, f"generations_{i}.png")
        vutils.save_image(all_images_resized, save_path_resized, normalize=True, nrow=8)

        # # Concatenate real and generated images
        # all_images = torch.cat((real_images, decoded_samples), dim=0)

        # # Save concatenated samples
        # save_path = os.path.join(imgs_dir, f"generations_{i}.png")
        # vutils.save_image(all_images, save_path, normalize=True, nrow=8)
    if i % 100 == 0:
        with torch.no_grad():
            real_energy = ebm(real_samples).mean()
            sample_energy = ebm(model_samples).mean()
        print(f'Iter {i} | Real Energy: {real_energy} | Sample Energy: {sample_energy}', flush=True)
    if i % 5000 == 0 or i == (n_iters - 1):
        torch.save(ebm.state_dict(), os.path.join(weights_dir, f"ebm_weights_{i}.pt"))
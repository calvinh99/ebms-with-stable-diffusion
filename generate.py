import os
import yaml
import torch
import argparse
import torchvision.utils as vutils
import importlib
from glob import glob
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description='Generation Configuration')
parser.add_argument('config_path', type=str, help='Path to the config.yaml file')
parser.add_argument('checkpoint_iter', type=int, help='Which version of the checkpoint to use')
args = parser.parse_args()

# Load configuration
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract experiment name and log directory
experiment_name = os.path.splitext(os.path.basename(args.config_path))[0]
log_dir = os.path.join(config['log_root'], experiment_name, 'weights')

# Initialize CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-2", subfolder="vae")
vae.to(device)

# Load EBM architecture from the latest checkpoint
ebm_checkpoint_path = os.path.join(log_dir, f"ebm_weights_{args.checkpoint_iter}.pt")
print(f"Loading from checkpoint: {ebm_checkpoint_path}")
model_module = importlib.import_module(config['ebm_module'])
create_model_from_config_fn = getattr(model_module, 'create_model_from_config')
ebm = create_model_from_config_fn(config)
ebm.load_state_dict(torch.load(ebm_checkpoint_path))
ebm = ebm.to(device)
ebm.eval()

# Initialize hyperparams
K = config['K']
m = 16 # less
n_channels = 4 # TODO: put in config
latent_width = 32  # Set this according to your specific model
latent_height = 32  # Set this according to your specific model
dtype = torch.float32

# MCMC sampling function
def sample_from_model(ebm, n_steps, batch_size, n_channels, latent_width, latent_height, dtype):
    samples = torch.randn(batch_size, n_channels, latent_width, latent_height,
                          device=device, dtype=dtype, requires_grad=True)
    for i in range(n_steps):
        print(f"Step: {i}")
        ebm_scores = ebm(samples)
        print(f"Initial distribution shape: {samples.shape}")
        print(f"Ebm scores shape: {ebm_scores.shape}")
        ebm_sum = ebm_scores.sum()
        print(f"Ebm sum: {ebm_sum}")
        gradient = torch.autograd.grad(ebm_sum, [samples], retain_graph=True)[0]
        print(f"Gradient shape: {gradient.shape}")
        samples.data += gradient + 1e-2 * torch.randn_like(samples)
    return samples.detach()

# Generate samples
generated_samples = sample_from_model(ebm, K, m, n_channels, latent_width, latent_height, dtype)
print(f"Generated latents shape: {generated_samples.shape}")
decoded_samples = []
for i in range(m):
    with torch.no_grad():
        decoded_sample = vae.decode(generated_samples[i].unsqueeze(0)).sample
    decoded_samples.append(decoded_sample)
decoded_samples = torch.cat(decoded_samples, dim=0)
print(f"Decoded samples shape: {decoded_samples.shape}")

# Save generated samples
# Visualize a batch of images using Matplotlib
def normalize_tensor(t, a, b):
    min_t = torch.min(t)  # Find current minimum
    max_t = torch.max(t)  # Find current maximum
    t = t - min_t         # Shift to start from zero
    t = t / (max_t - min_t)  # Scale to [0, 1] range
    t = t * (b - a)       # Scale to [a, b] range
    t = t + a             # Shift to start from 'a'
    return t

def plot_images(image_tensor, save_path):
    n_images = image_tensor.shape[0]
    n_cols = min(4, n_images)
    n_rows = (n_images // n_cols) + int(n_images % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    plt.subplots_adjust(wspace=0.02, hspace=0.05) # reduce gaps btwn images
    for i, ax in enumerate(axes.flat):
        img = normalize_tensor(image_tensor[i].cpu(), 0 ,1)
        img = img.permute(1, 2, 0).numpy()  # Preprocess for display
        ax.imshow(img)
        ax.axis('off')
    plt.savefig(save_path) # necessary for remote terminal execution

save_path = os.path.join(config['log_root'], experiment_name, f"eval_generations_{args.checkpoint_iter}.png")
# vutils.save_image(decoded_samples, save_path, normalize=True, nrow=4) # high res saving
plot_images(decoded_samples, save_path)
print(f"Generated samples saved to {save_path}")
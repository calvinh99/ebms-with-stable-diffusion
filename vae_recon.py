# Import necessary libraries
import torch
import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import seaborn as sns

def normalize_tensor(t, a, b):
    min_t = torch.min(t)  # Find current minimum
    max_t = torch.max(t)  # Find current maximum
    t = t - min_t         # Shift to start from zero
    t = t / (max_t - min_t)  # Scale to [0, 1] range
    t = t * (b - a)       # Scale to [a, b] range
    t = t + a             # Shift to start from 'a'
    return t

# Prepare CIFAR10 data with transformations
def prepare_data():
    transform = transforms.Compose([
        transforms.Resize(512), # Resizing the image to 512x512
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing pixel values to -1 to 1
    ])
    dataset = datasets.CIFAR10(root='./data', transform=transform, download=True)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Encode images to latent variables
def encode_images(vae, images):
    latents = []
    for image in images:
        with torch.no_grad():
            latent = vae.encode(image.unsqueeze(0)).latent_dist.sample()  # Sampling from latent distribution
        latents.append(latent)
    return torch.cat(latents)  # Concatenate all latent vectors

# Decode latent variables back to images
def decode_latents(vae, latents):
    decoded = []
    for latent in latents:
        with torch.no_grad():
            image = vae.decode(latent.unsqueeze(0)).sample  # Generate image from latent
        decoded.append(image)
    return torch.cat(decoded)  # Concatenate all generated images

# Calculate FID (Fréchet Inception Distance)
def calculate_fid(activations1, activations2):
    # Mean and covariance of activations
    mu1, sigma1 = np.mean(activations1, axis=0), np.cov(activations1, rowvar=False)
    mu2, sigma2 = np.mean(activations2, axis=0), np.cov(activations2, rowvar=False)
    # Compute FID based on squared mean diff and trace of covariance matrices
    squared_mean_diff = np.sum((mu1 - mu2)**2)
    cov_term = sigma1 + sigma2 - 2.0 * sqrtm(sigma1 @ sigma2, disp=False)[0]
    return np.real(np.trace(squared_mean_diff + cov_term))

# Calculate SSIM (Structural Similarity Index)
def calculate_ssim(images1, images2):
    images1, images2 = img_as_float(images1), img_as_float(images2)  # Convert to float for calculation
    ssim_values = []
    for img1, img2 in zip(images1, images2):
        # print(f'Inside calculate_ssim, shape of img1: {img1.shape}, shape of img2: {img2.shape}')
        # print(f"Min and Max of img1: {img1.min()}, {img1.max()}")
        # print(f"Min and Max of img2: {img2.min()}, {img2.max()}")
        ssim_value = ssim(img1, img2, multichannel=True, channel_axis=0, data_range=2.0)  # Multichannel for color images
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)  # Take average SSIM over all pairs

# Visualize a batch of images using Matplotlib
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

# Main script execution
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader = prepare_data()
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-2", subfolder="vae")
    vae = vae.to(device)
    # inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)  # Pre-trained Inception model for FID
    # inception_model = inception_model.to(device) # TODO: check whether we use .eval()

    for batch_images, _ in data_loader:
        batch_images = batch_images.to(device)
        print(f"Batch image shape: {batch_images.shape}")
        latents = encode_images(vae, batch_images)  # Encoding to latent space
        print(f"Latents shape: {latents.shape}")
        reconstructed_images = decode_latents(vae, latents)  # Decoding back to image
        reconstructed_images = normalize_tensor(reconstructed_images, -1, 1) # Normalize to same range as batch_images
        print(f"Reconstructed images shape: {batch_images.shape}")

        plot_images(batch_images, 'imgs/cifar10_sdvae_raw.png')
        plot_images(reconstructed_images, 'imgs/cifar10_sdvae_recon.png')

        # Convert PyTorch tensor to NumPy array for metrics calculation
        original_images_np = batch_images.cpu().numpy()
        reconstructed_images_np = reconstructed_images.cpu().numpy()

        ssim_score = calculate_ssim(original_images_np, reconstructed_images_np)  # Calculate SSIM
        print(f'SSIM score: {ssim_score}')

        # fid_score = calculate_fid(
        #     inception_model(batch_images).logits.detach().cpu().numpy(),
        #     inception_model(reconstructed_images).logits.detach().cpu().numpy()
        # )  # Calculate FID
        # print(f'FID score: {fid_score}')

        break
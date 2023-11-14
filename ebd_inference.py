import os
import yaml
import importlib
import torch
from diffusers import AutoencoderKL


# Load configuration
ebm_config_path = "configs/256cifar-7.yaml"
ebm_checkpoint_id = "99999"
with open(ebm_config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract experiment name and log directory
experiment_name = os.path.splitext(os.path.basename(ebm_config_path))[0]
log_dir = os.path.join(config['log_root'], experiment_name, 'weights')

# Load VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-2", subfolder="vae").to("cuda")

# Load EBM architecture from checkpoint
ebm_checkpoint_path = os.path.join(log_dir, f"ebm_weights_{ebm_checkpoint_id}.pt")
create_model_fn = getattr(importlib.import_module(config['ebm_module']), 'create_model_from_config')
ebm = create_model_fn(config)
ebm.load_state_dict(torch.load(ebm_checkpoint_path))
ebm = ebm.to("cuda")

# Initialize hyperparams
m = 1
K = config['K']
n_channels, latent_width, latent_height = 4, 32, 32
dtype = torch.float32

# MCMC sampling function
def sample_from_model(ebm, n_steps, batch_size, n_channels, width, height, dtype):
    samples = torch.randn(batch_size, n_channels, width, height,
                        device="cuda", dtype=dtype, requires_grad=True)
    for i in range(n_steps):
        ebm_scores = ebm(samples)
        gradient = torch.autograd.grad(ebm_scores.sum(), [samples], retain_graph=True)[0]
        samples.data += gradient + 1e-2 * torch.randn_like(samples)
    return samples.detach()

# Generate Image
def generate_image():
    start_mem = torch.cuda.memory_allocated()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    generated_samples = sample_from_model(ebm, K, m, n_channels, latent_width, latent_height, dtype)
    decoded_samples = []
    for i in range(m):
        with torch.no_grad():
            decoded_samples.append(vae.decode(generated_samples[i].unsqueeze(0)).sample)
    end_time.record()
    
    torch.cuda.synchronize()  # Wait for the events to be recorded
    time_elapsed = start_time.elapsed_time(end_time)
    mem_used = torch.cuda.memory_allocated() - start_mem

    return torch.cat(decoded_samples, dim=0), time_elapsed, mem_used

# Profile the sampling process
n_runs = 100
total_time = 0
total_memory = 0

for _ in range(n_runs):
    _, time_elapsed, mem_used = generate_image()
    total_time += time_elapsed
    total_memory += mem_used

# Calculate average time and memory
avg_time = total_time / n_runs
avg_memory = total_memory / n_runs  # Memory in bytes

print(f"Average inference time over {n_runs} runs: {avg_time / 1000} seconds")
print(f"Average memory used over {n_runs} runs: {avg_memory / (1024**2)} MiB")
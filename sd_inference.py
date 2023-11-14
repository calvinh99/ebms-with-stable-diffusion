import timeit
import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion Pipeline:
#     Text Encoder                 (prompt -> embeddings)
#     Latent Diffusion Model       (embeddings + noise -> latent representation)
#     VAE Decoder                  (latent representation -> final image)
#     Safety Checkers              (final image -> safe or not)
model_id = "CompVis/stable-diffusion-v1-2"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

# Function to generate an image and profile GPU memory usage
def generate_image():
    text_input = "Energy"

    with torch.no_grad():
        start_mem = torch.cuda.memory_allocated()
        image = sd_pipeline(text_input).images[0]
        end_mem = torch.cuda.memory_allocated()
        mem_used = end_mem - start_mem

    return mem_used, image

# Profile inference time and memory usage
n_runs = 100
total_time = 0
total_memory = 0
for _ in range(n_runs):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    mem_used, image = generate_image()
    end_time.record()
    
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    total_time += start_time.elapsed_time(end_time)
    total_memory += mem_used

# Calculate average time and memory
avg_time = total_time / n_runs
avg_memory = total_memory / n_runs  # Memory in bytes

print(f"Average inference time over {n_runs} runs: {avg_time / 1000} seconds")
print(f"Average memory used over {n_runs} runs: {avg_memory / (1024**2)} MiB")

# Save the last generated image
image_save_path = "imgs/from_sd_text_input_energy.jpg"
image.save(image_save_path)
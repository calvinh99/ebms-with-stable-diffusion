# Energy Based Diffusion

I use energy based models to learn the latent space of stable diffusion's pretrained variational autoencoder. This sounds fancy, but basically I train a neural network to output high scores for matrices that are similar to the target matrices. The target matrices are the latent codes produced by taking a target image and then sampling a matrix from the distribution outputted by the encoder half of stable diffusion's vae.

What latent codes are:<br>
Image -> Vae's Encoder -> Distribtuion<br>
Distribution -> Random Sampling -> Latent Codes (Training data)<br>

How we calculate loss:<br>
What we generated -> Energy based model -> Score 1<br>
Training data -> Energy based model -> Score 2<br>
Loss = Score 1 - Score 2<br>

We basically try to minimize difference between scores, so that the energy based model can give good score for a good generation.

Now this means the encoding part of image generation just involves using the trained energy based model in some langevin dynamics sampling for around 100 iterations. This is super duper fast, a speed up of **50x** and doesn't even need a GPU (the decoder still does).


*In the works to write a much more low-level explanation, I also dislike high-level abstractions haha...*


## Inference Performance

The inference times reported in this project were obtained using the following hardware configuration:

**GPU**
- **Type**: NVIDIA GeForce RTX 4090
- **Driver Version**: 525.116.04
- **CUDA Version**: 12.0
- **Memory**: 24564 MiB

**CPU**
- **Model**: Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
- **Architecture**: x86_64
- **CPU(s)**: 20
- **Thread(s) per core**: 2
- **Core(s) per socket**: 10
- **Socket(s)**: 1
- **NUMA node(s)**: 1

**RAM**
- **Total RAM**: 110 GiB

### Image Generation with pure Stable Diffusion
The script to run is `sd_inference.py`
```
Average inference time over 100 runs: 3.1468422729492187 seconds
Average memory used over 100 runs: 0.0837548828125 MiB
```

### Image Generation with Energy Based Models + SD Decoder
The script to run is `ebd_inference.py`
```
Average inference time over 100 runs: 0.06323544548034668 seconds
Average memory used over 100 runs: 0.928125 MiB
```



Speed up of **50x** (the issue is that I'm not sure if the slowness of StableDiffusionPipeline is caused by the text encoder + latent diffusion model or by the safety checks, it's most likely the prior, which means our results are valid).

One limitation is no "text prompting". However, what if we trained our ebms on a huge anime catgirl dataset? We could generate completely new catgirls on the fly at lightning speeds. In addition, the batch size of the output can be scaled up to memory limits -> which means we can generate millions every second.
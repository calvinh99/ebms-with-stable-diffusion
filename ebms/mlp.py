import torch.nn as nn

class EnergyFunction(nn.Module):
    def __init__(self, l=0.2):
        super(EnergyFunction, self).__init__()
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*64*64, 1024), # vae encodes 3, 512, 512 to 4, 64, 64 latents
            nn.LeakyReLU(l),
            nn.Linear(1024, 256),
            nn.LeakyReLU(l),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.f(x).squeeze()
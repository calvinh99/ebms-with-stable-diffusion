import torch.nn as nn

class EnergyFunction(nn.Module):
    def __init__(self):
        super(EnergyFunction, self).__init__()

        self.f = nn.Sequential(
            # Input: (4 x 64 x 64)
            nn.Conv2d(4, 64, 3, 1, 1),  # Output: (64 x 64 x 64)
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),  # Output: (128 x 32 x 32)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # Output: (256 x 16 x 16)
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),  # Output: (512 x 8 x 8)
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 4, 2, 1),  # Output: (1024 x 4 x 4)
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, 4, 1, 0)  # Output: (1 x 1 x 1)
        )

    def forward(self, x):
        return self.f(x).squeeze()
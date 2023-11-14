import torch.nn as nn

class EnergyFunction(nn.Module):
    def __init__(self, conv_layer_params, fc_layer_params, l=0.2):
        super(EnergyFunction, self).__init__()
        conv_layers = []
        for params in conv_layer_params:
            conv_layers.append(nn.Conv2d(*params))
            conv_layers.append(nn.LeakyReLU(l))
        
        fc_layers = [nn.Flatten()]
        for params in fc_layer_params:
            fc_layers.append(nn.Linear(*params))
            fc_layers.append(nn.LeakyReLU(l))

        self.f = nn.Sequential(*(conv_layers + fc_layers[:-1]))

    def forward(self, x):
        return self.f(x).squeeze()

def create_model_from_config(config):
    return EnergyFunction(config['conv_layers'], 
                          config['fc_layers'])
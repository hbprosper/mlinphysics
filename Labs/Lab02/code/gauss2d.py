import torch
import torch.nn as nn

class Gauss2D(nn.Module):

    def __init__(self, add_layernorm=False): 
        super().__init__()

        H = 15
        if add_layernorm:
            self.net = nn.Sequential(
                nn.Linear(2, H), nn.SiLU(), nn.LayerNorm(H),
                nn.Linear(H, H), nn.SiLU(), nn.LayerNorm(H),
                nn.Linear(H, H), nn.SiLU(), nn.LayerNorm(H),
                nn.Linear(H, 1), nn.Sigmoid())
        else:
            self.net = nn.Sequential(
                nn.Linear(2, H), nn.SiLU(), 
                nn.Linear(H, H), nn.SiLU(),
                nn.Linear(H, H), nn.SiLU(),
                nn.Linear(H, 1), nn.Sigmoid())

    def save(self, paramsfile):
        # save parameters of neural network
        torch.save(self.state_dict(), paramsfile)

    def load(self, paramsfile):
        # load parameters of neural network and set to eval mode
        self.load_state_dict(torch.load(paramsfile, 
                                        weights_only=True,
                                        map_location=torch.device('cpu')))
        self.eval()

    def forward(self, x):
        return self.net(x)

model = Gauss2D()

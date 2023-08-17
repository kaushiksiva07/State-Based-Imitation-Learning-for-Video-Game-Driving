import torch
import torch.nn.functional as F


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(11, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 256)
        self.layer4 = torch.nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'state_agent/model.th'))


def load_model():
    from torch import load
    from os import path
    r = Network()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'state_agent/model.th'), map_location='cpu'))
    return r

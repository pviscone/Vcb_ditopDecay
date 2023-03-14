import torch

class MLP(torch.nn.Module):
    def __init__(self, out_activation=None, arch=None):
        super().__init__()

        assert arch is not None
        assert out_activation is not None
        
        self.layers = torch.nn.Sequential()
        for i, l_minus1, l_plus1 in enumerate(zip(arch[:-1], arch[1:])):
            self.layers.append(torch.nn.Linear(l_minus1, l_plus1))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)
            if i != len(arch):
                self.layers.append(torch.nn.ReLU())
            else:
                self.layers.append(out_activation)

    def forward(self, x):
        return self.layers(x)

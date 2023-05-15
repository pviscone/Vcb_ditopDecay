import torch

class MLP(torch.nn.Module):
    def __init__(self, out_activation=None, arch=None,dropout=0.15):
        super().__init__()
        assert arch is not None
        assert out_activation is not None

        self.layers = torch.nn.Sequential()
        for i, pair in enumerate(zip(arch[:-1], arch[1:])):
            l_minus1, l_plus1 =pair
            
            self.layers.append(torch.nn.Linear(l_minus1, l_plus1))
            torch.nn.init.kaiming_normal_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)
            
            if i != len(arch)-2:
                self.layers.append(torch.nn.SiLU())
            else:
                self.layers.append(out_activation)
            
            if (    i % 2 == 0
                    and (i<len(arch)-2
                    or (i==len(arch)-2 and type(out_activation) != torch.nn.LogSoftmax))
                    and (dropout is not None)
                ):
                self.layers.append(torch.nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)
    


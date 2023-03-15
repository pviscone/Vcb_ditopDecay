import torch
from MLP import MLP

class Attention(torch.nn.Module):
    def __init__(self, input_dim=None,mlp_arch=None, n_heads=1,dropout=0.15):
        super().__init__()

        assert mlp_arch is not None
        assert input_dim is not None
        mlp_arch.insert(0,input_dim)

        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=n_heads,batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.mlp=MLP(arch=mlp_arch, out_activation=torch.nn.LeakyReLU(0.1))
        

    def forward(self, x):
        out = self.norm1(x)
        out, _ = self.attention(out, out, out,need_weights=False)
        out=self.dropout(out)
        out=self.norm2(out+x)
        out=self.mlp(out)
        return out

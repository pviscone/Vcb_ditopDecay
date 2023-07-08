import torch
from torch import nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.norm=nn.LayerNorm(input_dim)
        
    def forward(self, batch_rep,pad_mask=None):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        if pad_mask is None:
            pad_mask=0
        
        softmax = nn.functional.softmax
        
        att_w = softmax(self.W(batch_rep).squeeze(-1)+pad_mask,dim=1).unsqueeze(-1)
        res = torch.sum(batch_rep * att_w, dim=1)
        res=self.norm(res)

        return res
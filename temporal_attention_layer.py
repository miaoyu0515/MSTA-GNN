# _*_ coding: utf-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

device=torch.device('cuda')
# device=torch.device('cpu')

class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.vff = nn.Linear(outfea, outfea)
        self.conv1 = nn.Conv1d(outfea, outfea, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(outfea, outfea, kernel_size=3, padding=1, bias=True)
        # self.conv1=nn.Conv2d(12,12,(1,3),bias=True)
        # self.conv2=nn.Conv2d(12,12,(1,3),bias=True)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )
        self.d = 2

    def forward(self, x,score_his=None):# x : b t n hidden
        # b, t, n, c = x.shape
        b, t, c = x.shape
        x_permuted = x.permute(0, 2, 1)

        query=self.conv1(x_permuted)
        key=self.conv2(x_permuted)
        value=self.vff(x)

        # 以下操作为维度顺序调整
        query = query.permute(0, 2, 1)
        key = key.permute(0, 2, 1)
        # value = value.permute(0, 2, 1, 3)

        A = torch.matmul(query, key.transpose(1, 2))
        A /= (c ** 0.5)
        A = torch.softmax(A, -1)

        # res attention
        if score_his is not None:
            try:
                A=A+score_his
            except:
                pass
        score_his=A.clone().detach()

        value = torch.matmul(A, value)
        # value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x),score_his


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=12):
        super(PositionalEncoding, self).__init__()


    def forward(self, x):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(x.size(1), x.size(2)).to(device)
        position = torch.arange(0, x.size(1)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.size(2), 2) *
                             -(math.log(10000.0) / x.size(2)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,T,F]
        self.register_buffer('pe', pe)
        # print(x.size())
        # print(pe.size())
        x = x + Variable(pe,
                         requires_grad=False)
        return x

class TA_layer(nn.Module):
    def __init__(self,dim_in,dim_out,num_layer,d=2,att_his=False):
        super(TA_layer,self).__init__()
        # self.linear1=nn.Linear(dim_in,dim_out)
        self.trans_layers=nn.ModuleList(Transform(dim_out,d) for l in range(num_layer))
        self.PE=PositionalEncoding(dim_out)
        self.num_layer=num_layer
        self.att_his=att_his
        if att_his:
            self.score_his = torch.zeros((64, 12, 12), requires_grad=False).to(device)
    def forward(self, x):
        # x=self.linear1(x)
        x=self.PE(x)
        for l in range(self.num_layer):
            if  self.att_his:
                x,self.score_his=self.trans_layers[l](x,self.score_his)
            else:
                x,_=self.trans_layers[l](x)
        return x

if __name__=="__main__":
    x = torch.randn(32, 12, 170, 64)
    dim_in=64
    dim_out=64
    num_layer=2
    d=64
    TA=TA_layer(dim_in,dim_out,num_layer,d)
    res=TA(x)
    print(res.shape)
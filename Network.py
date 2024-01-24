import torch
import numpy as np
from torch import nn
from Arguments import Arguments
args = Arguments()

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim)                                   
            )
        
    def forward(self, x):
        y = self.network(x)
        # y[:,-1] = torch.sigmoid(y[:,-1])        
        return y

class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Mlp, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),       
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)                     
            )
        
    def forward(self, x):
        y = self.network(x) 
        return y

class FCN(nn.Module):
    """Fully Convolutional Network"""
    def __init__(self, n_channels, output_size):
        super(FCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size= (3, 3), padding='same'),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(n_channels, output_size, kernel_size= (3, 3)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
            )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y).squeeze()
        # y[:,-1] = torch.sigmoid(y[:,-1])
        return y


class ACN(nn.Module):
    """Adaptive Convolutional Network"""
    def __init__(self, n_channels, pooling_size, output_size):
        super(ACN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size= (3, 3), padding='same'),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(pooling_size)
            )
        self.layer2 = nn.Linear(n_channels*pooling_size[0]*pooling_size[1], output_size)

    def forward(self, x):
        y = self.layer1(x)
        y = y.flatten(1)
        y = self.layer2(y)
        # y[:,-1] = torch.sigmoid(y[:,-1])
        return y


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  #(batch, seq, feature)
        # self.classifier = nn.Linear(args.n_layers*hidden_size, output_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):        
        out, out1 = self.rnn(x)
        out = out[:,-1,:]
        # out = out.flatten(1)
        out = self.classifier(out)
        # out[:, -1] = torch.sigmoid(out[:, -1])
        return out


class Attention(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, 1)
        self.classifier = nn.Linear(args.n_layers*args.n_qubits, output_size)

    def forward(self, x):        #(batch, seq, feature)
        x = x.permute(1, 0, 2)   #(seq, batch, feature)
        out, _ = self.attention(x, x, x)
        out = out.permute(1, 0, 2)
        out = self.classifier(out.flatten(1))
        # out[:, -1] = torch.sigmoid(out[:, -1])
        return out

def positional_encoding(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * torch.div(i, 2, rounding_mode='floor')) / d_model)
    angle_rads = pos * angle_rates
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

def normalize(x):
    try:
        x = (x - torch.mean(x, dim=(1,2)).unsqueeze(-1).unsqueeze(-1)) / torch.std(x, dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        # x = (x - torch.mean(x)) / torch.std(x)
    except Exception as e:
        x = x
    return x
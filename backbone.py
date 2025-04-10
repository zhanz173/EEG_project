## EEG backbone model

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

# 1D postional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 1D Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, kernel_size:list, embeding_size:int):
        super(ConvEncoder, self).__init__()
        self.conv_block = []
        for i in range(len(kernel_size)):
            self.conv_block.append(ConvBlock(in_channels, embeding_size, kernel_size[i], (kernel_size[i]+1)//2, (kernel_size[i]-1)//2))
            in_channels = embeding_size
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)
    
#2D Convolutional block
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ConvEncoder2D(nn.Module):
    def __init__(self, in_channels, kernel_size:list, embeding_size:int, stride:int, padding:int):
        super(ConvEncoder2D, self).__init__()
        self.conv_block = []
        for i in range(len(kernel_size)):
            self.conv_block.append(ConvBlock2D(in_channels, embeding_size, kernel_size[i], stride, padding))
            in_channels = embeding_size
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)
    

class EEGTransformerEncoder(nn.Module):
    """
    EEG Transformer Encoder model
    
    Args:
    
    in_channels: int, number of input channels
    kernel_size: list, list of kernel sizes for convolutional blocks
    embeding_size: int, size of the embedding
    stride: int, stride for convolutional blocks
    padding: int, padding for convolutional blocks
    num_layers: int, number of transformer layers
    num_heads: int, number of attention heads
    hidden_size: int, hidden size of the transformer
    dropout: float, dropout rate
    
    Inputs: x
    - **x** (batch, time, dim): Tensor containing input sequence
    
    Returns: x
    - **x** (batch, time, dim): Tensor produces by EEG Transformer model
    """
    def __init__(self, in_channels=1, kernel_size=[3,3,3], embeding_size=32, num_layers=4, num_heads=8, hidden_size=64, dropout=0.1):
        super(EEGTransformerEncoder, self).__init__()
        self.conv_encoder = ConvEncoder(in_channels, kernel_size, embeding_size )
        encoder_layers = TransformerEncoderLayer(d_model=embeding_size, nhead=num_heads, dropout=0.1, dim_feedforward=hidden_size, batch_first=True, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        #self.positional_encoding = PositionalEncoding(embeding_size, max_len=256,dropout=0)


    def time_permute(self, x):
        #time shuffle
        batch_size, seq_len, num_channels = x.size()
        index_array = torch.zeros(batch_size, seq_len).to(x.device)
        idx = np.random.choice(seq_len, 2, replace=False)
        # exhange the two time points
        x[:, idx[0], :], x[:, idx[1], :] = x[:, idx[1], :], x[:, idx[0], :]
        index_array[idx] = 1
        
        return x, index_array

    def forward(self, x, permute=False):
        x = self.conv_encoder(x)
        x = x.permute(0, 2, 1) # (batch, dim, time) -> (batch, time, dim)
        if permute:
            x, index_array = self.time_permute(x)
        #x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            return x, index_array 
        else:
            x = self.transformer_encoder(x)
            return x
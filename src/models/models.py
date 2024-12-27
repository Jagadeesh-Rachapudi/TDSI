import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import gc
import os
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import Adam

# Encoder Module
class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, kernel_size=7, num_layers=4):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2
                )
            )
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Decoder Module
class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1, kernel_size=7, num_layers=4):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(
                nn.ConvTranspose1d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1
                )
            )
            if i < num_layers - 1:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# AudioSeal Watermarking Model
class AudioSealWM(nn.Module):
    def __init__(self, encoder, decoder, nbits=16, hidden_dim=128):
        super(AudioSealWM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(2, hidden_dim)  # Embedding for watermark bits

    def forward(self, audio, msg):
        # Encode audio
        encoded_audio = self.encoder(audio)

        # Embed watermark message
        msg_embeddings = self.embedding(msg)  # Shape: (batch_size, nbits, hidden_dim)
        msg_embeddings = msg_embeddings.mean(dim=1, keepdim=True)  # Average embedding, Shape: (batch_size, 1, hidden_dim)
    
        # Match dimensions for addition
        msg_embeddings = msg_embeddings.permute(0, 2, 1)  # Change to (batch_size, hidden_dim, 1)
        msg_embeddings = msg_embeddings.expand_as(encoded_audio)  # Expand to match encoded_audio shape
    
        # Combine encoded audio with message embedding
        combined = encoded_audio + msg_embeddings
    
        # Decode back to audio
        reconstructed_audio = self.decoder(combined)
        return reconstructed_audio

# Export statements
__all__ = ["Encoder", "Decoder", "AudioSealWM"]

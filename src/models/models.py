import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, kernel_size=7, num_layers=4):
        super(Encoder, self).__init__()
        self.conv_blocks = nn.ModuleList()
        in_channels = input_dim

        # Convolutional layers with down-sampling
        for _ in range(num_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))
            in_channels = hidden_dim

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        skip_connections = []
        for block in self.conv_blocks:
            x = block(x)
            skip_connections.append(x)  # Save skip connection
        x = x.transpose(1, 2)  # Convert to (batch, time, channels) for LSTM
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # Back to (batch, channels, time)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1, kernel_size=7, num_layers=4):
        super(Decoder, self).__init__()
        self.deconv_blocks = nn.ModuleList()

        # Transposed convolutional layers for up-sampling
        for _ in range(num_layers):
            self.deconv_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))

        # Final output layer
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x, skip_connections):
        for block, skip in zip(self.deconv_blocks, reversed(skip_connections)):
            x = block(x)
            if x.size() == skip.size():  # Ensure matching dimensions before addition
                x = x + skip  # Add skip connection
        x = self.output_layer(x)
        return x


class AudioRegenModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, kernel_size=7, num_layers=4):
        super(AudioRegenModel, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
        self.decoder = Decoder(hidden_dim=hidden_dim, output_dim=input_dim, kernel_size=kernel_size, num_layers=num_layers)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        regenerated = self.decoder(encoded, skip_connections)
        return regenerated

# Example usage
if __name__ == "__main__":
    model = AudioRegenModel(input_dim=1, hidden_dim=128, kernel_size=7, num_layers=4)
    dummy_input = torch.randn(1, 1, 64000)  # Batch size = 1, mono audio, 64000 samples (4 seconds at 16kHz)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

from torch import nn

class DoubleConvolution(nn.Module):
    """
    Double convolution block for the U-Net architecture.
    Compared to the original architecture BatchNorm and "same" padding have been added.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),           
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder_block(x)

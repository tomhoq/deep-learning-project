from torch import nn
from .double_conv import DoubleConv

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Encoder block for the U-Net architecture. 
        Consists of DoubleConv followed by a 2x2 max pooling layer (for downsampling).

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        """
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(2)


    def forward(self, x):
        """
        Returns the downsampled features and the intermediate features.
        """

        intermediate_features = self.conv(x)
        return self.downsample(intermediate_features), intermediate_features 
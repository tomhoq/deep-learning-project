from torch import cat
from torch import nn
from .double_conv import DoubleConv


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Decoder block for the U-Net architecture. Consists of an upsample layer followed by DoubleConv.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        """
        super().__init__()

        # Upsample layer. Output has channels halved but doubled resolution
        self.upscale = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Double convolution block
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_features):
        """
        :param x: Input tensor (from previous layer)
        :param skip_features: Features from the encoder part of the U-Net
        """

        upsampled_x = self.up(x)

        # Concatenate by column (in our version we do not need cropping since we use "same" padding)
        x = cat([upsampled_x, skip_features], dim=1)

        return self.conv(x)

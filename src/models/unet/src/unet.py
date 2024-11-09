from torch import nn
from torch.utils import checkpoint
from .modules import DoubleConv, DecoderBlock, EncoderBlock


class UNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        """
        Defines the U-Net architecture.

        :param n_channels: number of input channels
        :param n_classes: number of output channels
        """

        super().__init__()

        self.n_input_channels = input_channels
        self.n_output_classes = output_classes

        self.enc1 = EncoderBlock(input_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.encoder1 = DecoderBlock(1024, 512)
        self.encoder2 = DecoderBlock(512, 256)
        self.encoder3 = DecoderBlock(256, 128)
        self.encoder4 = DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, output_classes, kernel_size=1, padding="same")


    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        b = self.bottleneck(x)

        x = self.encoder1(b, s4)
        x = self.encoder2(x, s3)
        x = self.encoder3(x, s2)
        x = self.encoder4(x, s1)

        return self.out(x, )


    def use_checkpointing(self):
        self.enc1 = checkpoint(self.enc1)
        self.enc2 = checkpoint(self.enc2)
        self.enc3 = checkpoint(self.enc3)
        self.enc4 = checkpoint(self.enc4)
        self.bottleneck = checkpoint(self.bottleneck)
        self.encoder1 = checkpoint(self.encoder1)
        self.encoder2 = checkpoint(self.encoder2)
        self.encoder3 = checkpoint(self.encoder3)
        self.encoder4 = checkpoint(self.encoder4)
        self.out = checkpoint(self.out)

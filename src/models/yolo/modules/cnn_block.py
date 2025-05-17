from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Defines the CNN block used by YOLO:  Conv -> Batchnorm -> LeakyReLU
        (here adding batchnorm)
        """

        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
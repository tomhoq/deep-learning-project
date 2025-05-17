from torch import nn
import torch
from models.yolo.modules.cnn_block import CNNBlock


class YOLOBuilder():
    def __init__(self, architecture, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture
        self.in_channels = in_channels
    

    def create_conv_layers(self):
        layers = []
        in_channels = self.in_channels
        
        for x in self.architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            #
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            #
            elif type(x) == list:
                conv1 = x[0]    # tuple
                conv2 = x[1]    # tuple
                repeats = x[2]  # int
                
                for _ in range(repeats):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    

    def create_fully_connected_layers(self, S, B, C):
        """
        :param S: Number of cells in the grid (original paper 7x7 grid, S = 7)
        :param B: Number of boxes per cell (original paper B = 2)
        :param C: Number of classes (original paper C = 20)
        """
        return nn.Sequential(
                nn.Flatten(), 
                #
                nn.Linear(1024 * S * S, 4096),
                nn.LeakyReLU(0.1), 
                nn.Dropout(0.5), 
                #
                nn.Linear(4096, S * S * (C + B * 5))
            ) 

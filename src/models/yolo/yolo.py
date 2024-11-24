# Adjusted from https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch
from torch import nn
import torch
from models.yolo.modules.yolo_builder import YOLOBuilder


# Tuple: (kernel_size, number of filters, strides, padding)
# "M" = Max Pool Layer
# List: [(tuple), (tuple), <how many times to repeat>]
conv_layers_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class YOLO(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super().__init__()

        self.in_channels = in_channels
        self.builder = YOLOBuilder(conv_layers_config, in_channels)

        self.conv = self.builder.create_conv_layers()
        self.fc = self.builder.create_fully_connected_layers(S = 7, B = 2, C = num_classes)

        self._initialize_weights()


    @torch.no_grad()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.conv(x)
        return self.fc(torch.flatten(x, start_dim=1))       
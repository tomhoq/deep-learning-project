# From https://www.kaggle.com/code/iafoss/unet34-dice-0-87/notebook

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from models.resnet34.resnet34 import get_resnet34


RESNET34_PATH = '/zhome/82/4/212615/deep-learning-project/job_out/resnet34/23232852/model.pt'


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))
    

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    

class Unet34(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = get_resnet34()
        cut = -2
        layers = list(resnet.children())[:cut] 
        resnet = nn.Sequential(*layers)

        model_path = Path(RESNET34_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(str(model_path), map_location=device, weights_only=False)
        resnet.load_state_dict(state['model'], strict=False)

        self.rn = resnet.to(device)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]

    def freeze_resnet(self):
        """Freeze all parameters in the ResNet backbone."""
        print("[!] ResNet34 Backbone Freezed")
        for param in self.rn.parameters():
            param.requires_grad = False
    
    def close(self):
        for sf in self.sfs: sf.remove()

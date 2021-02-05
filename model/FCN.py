
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import numpy as np


__all__ = ["FCN"]

pre_model_path = "F:\\ez\pretrained_model\\vgg16-397923af.pth"


class FCN32VGG(nn.Module):
    def __init__(self, classes, pretrained=True):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(pre_model_path))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore = nn.ConvTranspose2d(classes, classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(classes, classes, 64))

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN32VGG(classes=19).to(device)
    summary(model, (3, 512, 1024))

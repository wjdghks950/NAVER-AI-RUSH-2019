import torch
import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, model_size, out_size):
        super().__init__()
        if model_size == 34:
            model = models.resnet34(pretrained=True)
            expansion = 1
        elif model_size == 101:
            model = models.resnet101(pretrained=True)
            expansion = 4
        elif model_size == 50:
            model = models.resnet50(pretrained=True)
            expansion = 1
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512 * expansion, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        x = self.net(image).squeeze(-1).squeeze(-1)
        noise=True
        if noise:
            noise = x.data.new(x.size()).normal_(0, 0.01)
            x = x + noise
        return x
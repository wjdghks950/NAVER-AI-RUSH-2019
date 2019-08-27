import torch
import torch.nn as nn
import torch.nn.functional as F

class custom_model(nn.Module):
    def __init__(self, num_classes=1):
        super(custom_model, self).__init__()
        self.num_classes = num_classes
        
        self.eif_net = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 100),
        )

        self.ff_net = nn.Sequential(
            nn.Linear(35, 60),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(60, 10),
        )

        self.classifier = nn.Linear(110, 1)

    def forward(self, eif, ff):
        _eif=self.eif_net(eif)
        _ff=self.ff_net(ff)
        all_features = torch.cat((_eif, _ff), 1)
        output = self.classifier(all_features)
        output = torch.sigmoid(output)
        return output
class custom_model2(nn.Module):
    def __init__(self, num_classes=1):
        super(custom_model2, self).__init__()
        self.num_classes = num_classes
        image_model = models.resnet18(pretrained=True)
        image_model = list(image_model.children())[:-1]
        image_model.append(nn.Conv2d(512, 256, 1))
        self.image_net = nn.Sequential(*image_model)
        
        self.eif_net = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 100),
        )

        self.ff_net = nn.Sequential(
            nn.Linear(35, 60),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(60, 10),
        )

        self.classifier = nn.Linear(356, 1)

    def forward(self, image, eif, ff):
        _img=self.image_net(image).squeeze(-1).squeeze(-1)
        _eif=self.eif_net(eif)
        _ff=self.ff_net(ff)
        #_image = torch.cat((_img, _eif), 1)
        #all_features = torch.cat((_image, _ff), 1)
        all_features = torch.cat((_img, _eif), 1)
        output = torch.sigmoid(self.classifier(all_features))
        #output = self.classifier(all_features)
        return output

class custom_model3(nn.Module):
    def __init__(self, num_classes=1):
        super(custom_model3, self).__init__()
        self.num_classes = num_classes
        
        self.eif_net = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 100),
        )

        self.ff_net = nn.Sequential(
            nn.Linear(35, 60),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(60, 10),
        )

        self.history_net = nn.Sequential(
            nn.Linear(2427, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 100),
        )

        self.classifier = nn.Linear(210, 1)

    def forward(self, eif, ff, sequence):
        _eif=self.eif_net(eif)
        _ff=self.ff_net(ff)
        _history = self.history_net(sequence)
        _features = torch.cat((_eif, _ff), 1)
        all_features = torch.cat((_features, _history),1)
        output = self.classifier(all_features)
        return output

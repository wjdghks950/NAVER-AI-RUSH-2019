import torch
import torch.nn as nn

class CTRHistoryModel(nn.Module):
    def __init__(self, max_len, num_classes=1, hidden_size=2048, batch_size=32, num_layers=2):
        self.num_classes = num_classes
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        image_model = models.resnet101(pretrained=True)
        image_model = list(image_model.children())[:-1]
        image_model.append(nn.Conv2d(2048, 64, 1))
        self.image_net = nn.Sequential(*image_model)
        
        # Sequence data - history
        self.seq_net = nn.LSTM(input_size=self.max_len, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.eif_net = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 100),
        )

        self.ff_net = nn.Sequential(
            nn.Linear(35, 60),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(60, 10),
        )

        self.classifier = nn.Linear(output_size, 1) # TODO: output_size = _img + _eif + _ff + _hist_out (size of all these concatenated)

        # TODO: initialize method - which one?

    def forward(self, image, eif, ff, history, hn, cn):
        _img = self.image_net(image).squeeze(-1).squeeze(-1)
        _eif = self.eif_net(eif)
        _ff = self.ff_net(ff)
        _hist_out, (hn, cn) = self.seq_net(history, (hn, cn))
        _features = torch.cat((_eif, _ff), 1)
        _features = torch.cat((_hist_out, _features), 1)
        output = torch.sigmoid(self.classifier(_features))


class custom_model(nn.Module):
    def __init__(self, num_classes=1):
        super(custom_model, self).__init__()
        self.num_classes = num_classes
        
        self.eif_net = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 100),
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
        image_model = models.resnet101(pretrained=True)
        image_model = list(image_model.children())[:-1]
        image_model.append(nn.Conv2d(2048, 64, 1))
        self.image_net = nn.Sequential(*image_model)

        self.eif_net = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 100),
        )

        self.ff_net = nn.Sequential(
            nn.Linear(35, 60),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(60, 10),
        )

        self.classifier = nn.Linear(174, 1)

    def forward(self, image, eif, ff):
        _img=self.image_net(image).squeeze(-1).squeeze(-1)
        _eif=self.eif_net(eif)
        _ff=self.ff_net(ff)
        _image = torch.cat((_img, _eif), 1)
        all_features = torch.cat((_image, _ff), 1)
        output = torch.sigmoid(self.classifier(all_features))

        return output

# example model and code
class MLP_only_flatfeatures(nn.Module):
    def __init__(self, num_classes=1):
        super(MLP_only_flatfeatures, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2083, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

        self._initialize_weights()

    def forward(self, extracted_image_feature, flat_features):
        x = torch.cat((extracted_image_feature, flat_features), 1)
        x = self.classifier(x)
        # x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
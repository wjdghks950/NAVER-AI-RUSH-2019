import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CTRHistoryModel(nn.Module):
    def __init__(self, max_len, num_classes=1, hidden_size=1024, batch_size=32, num_layers=2):
        super(CTRHistoryModel, self).__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.linear_out = 128

        image_model = models.resnet101(pretrained=True)
        image_model = list(image_model.children())[:-1]
        image_model.append(nn.Conv2d(2048, 64, 1))
        self.image_net = nn.Sequential(*image_model)
        
        # Sequence data - history
        self.seq_net = nn.LSTM(input_size=self.max_len, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.linear_out)

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

        output_size = 100 + 10 + self.linear_out

        self.classifier = nn.Linear(output_size, self.num_classes) # output_size = _img + _eif + _ff + _hist_out (size of all these concatenated)

    def init_hidden(self):
        # Initialize hidden and cell
        h0 = torch.zeros(num_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(num_layers, self.batch_size, self.hidden_size)

        return (h0, c0) # returns a tuple of hidden and cell_state

    def forward(self, image, eif, ff, history, hn, cn):
        _img = self.image_net(image).squeeze(-1).squeeze(-1)
        _eif = self.eif_net(eif)
        _ff = self.ff_net(ff)
        _hist_out, self.hidden = self.seq_net(history.view(len(history)), self.batch_size, -1)
        _hist_out = self.linear(_hist_out[0])
        _features = torch.cat((_eif, _ff), 1)
        _features = torch.cat((_hist_out, _features), 1)
        output = torch.sigmoid(self.classifier(_features))

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

        self.classifier = nn.Linear(110, self.num_classes)

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

        self.classifier = nn.Linear(356, self.num_classes)

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

        self.classifier = nn.Linear(210, self.num_classes)

    def forward(self, eif, ff, sequence):
        # print('[eif]:', eif)
        # print('[ff]:', ff)
        # print('[history]:', sequence)
        _eif=self.eif_net(eif)
        _ff=self.ff_net(ff)
        _history = self.history_net(sequence)
        _features = torch.cat((_eif, _ff), 1)
        all_features = torch.cat((_features, _history),1)
        output = self.classifier(all_features)
        return output

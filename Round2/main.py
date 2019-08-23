from data_local_loader import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime

from data_loader import feed_infer
from evaluation import evaluation_metrics
import nsml

# expected to be a difficult problem
# Gives other meta data (gender age, etc.) but it's hard to predict click through rate
# How to use image and search history seems to be the key to problem solving. Very important data
# Image processing is key. hint: A unique image can be much smaller than the number of data.
# For example, storing image features separately and stacking them first,
# then reading them and learning artificial neural networks is good in terms of GPU efficiency.
# -> image feature has been extracted and loaded separately.
# The retrieval history is how to preprocess the sequential data and train it on which model.
# Greatly needed efficient coding of CNN RNNs.
# You can also try to change the training data set itself. Because it deals with very imbalanced problems.
# Refactor to summarize from existing experiment code.

if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True

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
        output = self.classifier(all_features)

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

def get_custom(num_classes):
    return custom_model(num_classes=num_classes)
def get_custom2(num_classes):
    return custom_model2(num_classes=num_classes)
def get_mlp(num_classes):
    return MLP_only_flatfeatures(num_classes=num_classes)

def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.ckpt'))
        print('saved model checkpoints...!')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.ckpt'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded model checkpoints...!')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)


def _infer(root, phase, model, task):
    # root : csv file path
    print('_infer root - : ', root)
    with torch.no_grad():
        model.eval()
        test_loader, dataset_sizes = get_data_loader(root, phase)
        y_pred = []
        print('start infer')
        for i, data in enumerate(test_loader):
            images, extracted_image_features, labels, flat_features = data

            # images = images.cuda()
            extracted_image_features = extracted_image_features.cuda()
            flat_features = flat_features.cuda()
            # labels = labels.cuda()

            logits = model(extracted_image_features, flat_features)
            y_pred += logits.cpu().squeeze().numpy().tolist()

        print('end infer')
    return y_pred


def main(args):
    if args.arch == 'MLP':
        model = get_mlp(num_classes=args.num_classes)
    elif args.arch == 'custom':
        model = get_custom(num_classes=args.num_classes)
    elif args.arch == 'custom2':
        model = get_custom2(num_classes=args.num_classes)
        #maximum batch_size
        args.batch_size = 64

    if args.use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_nsml:
        bind_nsml(model, optimizer, args.task)
    if args.pause:
        nsml.paused(scope=locals())

    if (args.mode == 'train') or args.dry_run:
        train_loader, dataset_sizes = get_data_loader(
            root=os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data'),
            phase='train',
            batch_size=args.batch_size)

        start_time = datetime.datetime.now()
        iter_per_epoch = len(train_loader)
        best_loss = 1000
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')

        for epoch in range(args.num_epochs):
            for i, data in enumerate(train_loader):
                images, extracted_image_features, labels, flat_features = data
                #print(images.size())
                #B x [3 x 456 x 232] image
                #print(extracted_image_features.size())
                #B x 2048 feature_image
                #print(flat_features.size())
                #B x 35 

                if args.arch == 'custom2':
                    images = images.cuda()
                extracted_image_features = extracted_image_features.cuda()
                flat_features = flat_features.cuda()
                labels = labels.cuda()

                # forward
                if args.arch == 'MLP':
                    logits = model(extracted_image_features, flat_features)
                elif args.arch == 'Resnet':
                    logits = model(images, flat_features)
                elif args.arch == 'custom':
                    logits = model(extracted_image_features, flat_features)
                elif args.arch == 'custom2':
                    logits = model(images, extracted_image_features, flat_features)
                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(logits.squeeze(), labels.float()))

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    nsml.save('best_loss')  # this will save your best model on nsml.

                if i % args.print_every == 0:
                    elapsed = datetime.datetime.now() - start_time
                    print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
                          % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item()))
                if i % args.save_step_every == 0:
                    # print('debug ] save testing purpose')
                    nsml.save('step_' + str(i))  # this will save your current model on nsml.
            if epoch % args.save_epoch_every == 0:
                nsml.save('epoch_' + str(epoch))  # this will save your current model on nsml.
    nsml.save('final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)  # not work. check built_in_args in data_local_loader.py

    parser.add_argument('--train_path', type=str, default='train/train_data/train_data')
    parser.add_argument('--test_path', type=str, default='test/test_data/test_data')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((456, 232))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((456, 232))]')

    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=2)
    parser.add_argument('--save_step_every', type=int, default=1000)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="custom")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)

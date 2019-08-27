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
from model import custom_model, custom_model2, custom_model3, CTRHistoryModel

from data_loader import feed_infer
from evaluation import evaluation_metrics
from sklearn.metrics import f1_score
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

def get_custom(num_classes):
    return custom_model(num_classes=num_classes)
def get_custom2(num_classes):
    return custom_model2(num_classes=num_classes)
def get_custom3(num_classes):
    return custom_model3(num_classes=num_classes)
def get_history_model(num_classes):
    return CTRHistoryModel(max_len=512, num_classes=num_classes)

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
            images, extracted_image_features, labels, flat_features, sequence = data
            sequence = sequence.cuda()
            #images = images.cuda()
            extracted_image_features = extracted_image_features.cuda()
            flat_features = flat_features.cuda()
            # labels = labels.cuda()

            #logits = model(images, extracted_image_features, flat_features)
            logits = model(extracted_image_features, flat_features, sequence)
            logits = logits.cpu().squeeze().detach().numpy()
            logits = np.argmax(logits, axis=1)
            logits = logits.astype(float)
            y_pred += logits.tolist()

        print('end infer')
    return y_pred

def evaluation(y_true, y_pred):
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    score = f1_score(y_true=y_true, y_pred=y_pred, pos_label=1)
    return score.item()

def main(args):
    print(args)
    if args.arch == 'MLP':
        model = get_mlp(num_classes=args.num_classes)
    elif args.arch == 'custom':
        model = get_custom(num_classes=args.num_classes)
    elif args.arch == 'custom2':
        model = get_custom2(num_classes=args.num_classes)
        #maximum batch_size
        args.batch_size = 64
    elif args.arch == 'custom3':
        model = get_custom3(num_classes=args.num_classes)
    elif args.arch == 'history':
        model = get_history_model(num_classes=args.num_classes)

    if args.use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.arch == 'custom2' or args.arch == 'custom':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
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
            nsml.save('init')
        total_loss = 0
        for epoch in range(args.num_epochs):
            for i, data in enumerate(train_loader):
                images, extracted_image_features, labels, flat_features, sequence = data
                #print(images.size())
                #B x [3 x 456 x 232] image
                #print(extracted_image_features.size())
                #B x 2048 feature_image
                #print(flat_features.size())
                #B x 35 

                if args.arch == 'custom2':
                    images = images.cuda()
                if args.arch == 'custom3':
                    sequence = sequence.cuda()
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
                elif args.arch == 'custom3':
                    logits = model(extracted_image_features, flat_features, sequence)
                elif args.arch == 'history':
                    logist = model()
                criterion = nn.MSELoss()
                if args.arch == 'custom2' or args.arch == 'custom' or args.arch == 'custom3':
                    #weight = torch.tensor([0.06382, 1.])
                    weight = torch.tensor([0.06382, 2.])
                    weight = weight.cuda()
                    criterion = nn.CrossEntropyLoss(weight=weight)
                    loss = criterion(logits.squeeze(), labels.long().squeeze(-1))

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_pred = logits.cpu().squeeze().detach().numpy()
                y_true = labels.cpu().squeeze().detach().numpy()
                
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = y_pred.astype(float)
                y_true = y_true.astype(int)
                score = evaluation(y_true, y_pred)
                print('[ Training set [F1 score] ] : ', score)
                print('[ Training Loss ] : ', loss.item())
                total_loss +=loss.item()
                if i % args.print_every == 0 and i > 9:
                    total_loss = total_loss / args.print_every
                    if total_loss < best_loss:
                        best_loss = total_loss
                        nsml.save('best_loss')
                    elapsed = datetime.datetime.now() - start_time
                    print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
                          % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, total_loss))
                    total_loss = 0
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
    parser.add_argument('--use_read_history', type=bool, default=True)

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=2)
    parser.add_argument('--save_step_every', type=int, default=200)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="custom")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)

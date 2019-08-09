import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from resnet_model import Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader
from dataloader import AIRushDataset

#######MY CODE
import time
import math
def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))

def lr_scheduler(args, optimizer, epoch):
    lr = args.learning_rate * (0.5 ** ( epoch // 20 ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
######
def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        #input_size=128 # you can change this according to your model.
        input_size=224 # for RESNET34
        batch_size=200 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                      transform=transforms.Compose([
                                      transforms.Resize((input_size, input_size)),
                                      transforms.Normalize((0.8674, 0.8422, 0.8217), (0.2285, 0.2483, 0.2682)),
                                      transforms.ToTensor()])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        with torch.no_grad(): # No need for torch.backward() - no gradient calculation
            for batch_idx, image in enumerate(dataloader):
                image = image.to(device)
                output = model_nsml(image).double()
                
                output_prob = F.softmax(output, dim=1)
                predict = np.argmax(to_np(output_prob), axis=1)
                predict_list.append(predict)

        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector # this return type should be a numpy array which has shape of (138343, 1)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    
    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--resnet', default=True)
    parser.add_argument('--model_size', type=int, default=101)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350) # Fixed
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()
    print(args)
    start = time.time()
    torch.manual_seed(args.seed)
    device = args.device
    #recommended
    #from scratch
    #lr = 0.1
    #weight decay 1e-4
    #decrease lr when validation error plateaus...............;;;;;;;;;;;;;;
    if args.resnet:
        assert args.input_size == 224
        model = Resnet(args.model_size, args.output_size)
    else:
        model = Baseline(args.hidden_size, args.output_size)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() #multi-class classification task

    # model = model.to(device)
    # model.train()

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        model = model.to(device)
        model.train()
        # Warning: Do not load data before this line
        dataloader = train_dataloader(args.input_size, args.batch_size, args.num_workers)
        best_accuracy = 0
        best_checkpoint = 0
        for epoch_idx in range(1, args.epochs + 1):
            #It's not pretrained model. So start from high lr and adjust it by epoch.
            lr_scheduler(args, optimizer, epoch_idx)
            total_loss = 0
            total_correct = 0
            for batch_idx, (image, tags) in enumerate(dataloader): # Data augmentation happens in this line
                optimizer.zero_grad()
                image = image.to(device)
                tags = tags.to(device)
                output = model(image).double()
                loss = criterion(output, tags)
                loss.backward()
                optimizer.step()

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                if batch_idx % args.log_interval == 0:
                    print('Time : {}, Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(timeSince(start), batch_idx,
                                                                             len(dataloader),
                                                                             loss.item(),
                                                                             accuracy))
                total_loss += loss.item()
                total_correct += bool_vector.sum()

            accuracy = total_correct/len(dataloader.dataset)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_checkpoint = epoch_idx

            print('CHECKPOINT : {}, has Accuracy : {}'.format(str(epoch_idx), accuracy))
            print('BEST CHECKPOINT : {}, has BEST Accuracy : {}'.format(str(best_checkpoint), best_accuracy))
            nsml.save(epoch_idx)
            print('nsml model saved' + str(epoch_idx))
            print('Time : {}, Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(timeSince(start), epoch_idx,
                                                           args.epochs,
                                                           total_loss/len(dataloader.dataset),
                                                           total_correct/len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                "train__Loss": total_loss/len(dataloader.dataset),
                "train__Accuracy": total_correct/len(dataloader.dataset),
                })

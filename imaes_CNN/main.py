import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np

from torch.utils.data import DataLoader
from dataloader import CustomDataset
from model import cnn_model

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


def train(model, train_loader, optimizer):
    model.train()
    correct = 0

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    train_loss = 0
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        data, target = data.cuda(), target.cuda()

        data = data.to(DEVICE)
        target = torch.squeeze(target).to(DEVICE)

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()

        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    return train_loss, train_acc

def evaluate(model, test_loader,valid_loss_min):
    model.eval()
    correct = 0

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    val_loss = 0
    for batch_idx, samples in enumerate(test_loader):
        data, target = samples
        data, target = data.cuda(), target.cuda()

        data = data.to(DEVICE)
        target = torch.squeeze(target).to(DEVICE)

        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()

        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    val_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)

    if val_loss <= valid_loss_min:
        torch.save(model.state_dict(),'test_resnet18.pt.pt')
        valid_loss_min = val_loss

    return val_loss, val_acc

def test(model, test_loader):
    model.eval()

    for batch_idx, samples in enumerate(test_loader):
        data, target = samples
        data, target = data.cuda(), target.cuda()
        # print(data.shape, target.shape)

        data = data.to(DEVICE)
        target = torch.squeeze(target).to(DEVICE)

        output = model(data)

        prediction = output.max(1, keepdim=True)[1]
        print(prediction)
        if prediction==target.view_as(prediction):
            print('correct')
        else:
            print('wrong')

        # print('Batch Index: {}\tLoss: {:.6f}'.format(batch_idx, loss.item()))

def main(args):
    # train
    if args.mode == 'train':
        train_dataset = CustomDataset(mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)


        val_dataset = CustomDataset(mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=0)
        print(train_dataloader, val_dataloader)


        model = cnn_model('resnet18')
        model = model.to(DEVICE)

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        valid_loss_min = np.Inf
        acc_prev = 0.0
        for epoch in range(args.epoch):
            # train set
            loss, acc = train(model, train_dataloader, optimizer)
            # validate set
            val_loss, val_acc = evaluate(model, val_dataloader,valid_loss_min)

            print('Epoch:{}\tTrain Loss:{:.6f}\tTrain Acc:{:2.4f}'.format(epoch, loss, acc))
            print('Val Loss:{:.6f}\tVal Acc:{:2.4f}'.format(val_loss, val_acc))

            # scheduler update
            scheduler.step()
    else:
        test_dataset = CustomDataset(mode = 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = cnn_model('resnet18')

        device = torch.device('cpu')
        model.load_state_dict(torch.load('test_resnet18.pt', map_location=device))
        model = model.to(DEVICE)
        test(model, test_dataloader)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=1,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=30,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--shuffle',
        help='True, or False',
        default=True,
        type=bool)
    parser.add_argument(
        '--mode',
        help='train or evaluate',
        default='evaluate',
        type=str)

    args = parser.parse_args()
    print(args)

    main(args)
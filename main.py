'''Train CIFAR10 with PyTorch.'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from spatial_transforms import (Compose, ToTensor, FiveCrops, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, TenCrops, FlippedImagesTest, CenterCrop)

import torchvision
from makeDataset import *
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--seqLen', type=int, default=30, help='Length of sequence')
parser.add_argument('--trainBatchSize', type=int, default=2, help='Training batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data_path', type=str, default='./data/raw_frames/violentflow', help='Directory containing data sequences')

args = parser.parse_args()

def make_split(data_dir):
    Dataset = []
    Labels = []
    for target in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, target)
        if not os.path.isdir(d):
            continue
        if "violence" in d:
            Labels.append(1)
        else:
            Labels.append(0)
        Dataset.append(d)
    
    train_path, test_path, train_y, test_y =  train_test_split(Dataset,Labels, test_size=0.20, random_state=42) 
    # NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]

    
    # return Dataset, Labels, NumFrames
    return train_path, train_y, test_path, test_y

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data_path = args.data_path
seqLen=args.seqLen
testBatchSize=1

trainX, trainY, testX, testY = make_split(data_path)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
normalize = Normalize(mean=mean, std=std)
spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                             ToTensor(), normalize])

vidSeqTrain = makeDataset(trainX, trainY, spatial_transform=spatial_transform,
                                seqLen=seqLen)

trainLoader = torch.utils.data.DataLoader(vidSeqTrain, batch_size=args.trainBatchSize,
                            shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])

vidSeqTest = makeDataset(testX, testY, seqLen=seqLen,
    spatial_transform=test_spatial_transform)

testLoader = torch.utils.data.DataLoader(vidSeqTest, batch_size=testBatchSize,
                        shuffle=False, num_workers=1)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testLoader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG16')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()e
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    net = VGG_ATT(mode='dp')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, 50)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    scheduler.step()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

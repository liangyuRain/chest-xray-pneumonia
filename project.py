# -*- coding: utf-8 -*-
"""Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/liangyuRain/chest-xray-pneumonia/blob/master/Project.ipynb
"""

import torch
import torch.nn as nn
import numpy as np
import scipy
import os

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import torch.utils.data

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        
        output = model(data)

        # for handling tuple outputs as in Inception V3 model
        if isinstance(output, tuple):
            loss = sum((model.loss(o, label) for o in output))
            pred = output[0].max(1, keepdim=True)[1]
        else:
            loss = model.loss(output, label)
            pred = output.max(1, keepdim=True)[1]

        correct += pred.eq(label.view_as(pred)).sum().item()

        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print('Train Accuracy: {0}%'.format(int(train_accuracy)))
    return np.mean(losses), train_accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)

            if isinstance(output, tuple):
                loss = sum((model.testLoss(o, label) for o in output))
                pred = output[0].max(1, keepdim=True)[1]
            else:
                loss = model.testLoss(output, label)
                pred = output.max(1, keepdim=True)[1]

            test_loss += loss.item()
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_accuracy


LR = 1e-5
WEIGHT_DECAY = 0
MOMENTUM = 0.9
EPOCHS = 5
BATCH_SIZE = 32
USE_CUDA = True
SEED = 0
LOG_INTERVAL = 20
LAYERS_TO_TRAIN = 10
MODEL = 'INCEPT'
#'VGG' #'ALEX' #'RES' #'DENSE' #'INCEPT'
NUM_CLASSES = 2


use_cuda = USE_CUDA and torch.cuda.is_available()
torch.manual_seed(SEED)
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

import multiprocessing
print('num cpus:', multiprocessing.cpu_count())
kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}

resize_flag = True
input_size = (224, 224)
if MODEL == 'INCEPT':
    model = torchvision.models.inception_v3(pretrained=True)
    input_size = (299, 299)
elif MODEL == 'VGG':
    model = torchvision.models.vgg16(pretrained=True)        
elif MODEL == 'ALEX':
    model = torchvision.models.alexnet(pretrained=True)
elif MODEL == 'DENSE':
    model = torchvision.models.densenet161(pretrained=True)
elif MODEL == 'RES':
    model = torchvision.models.resnet18(pretrained=True)


# set categories to NUM_CLASSES (2 by default)
if MODEL == 'INCEPT' or MODEL == 'RES': 
    num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, NUM_CLASSES, bias=True)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES, bias=True)
    )
elif MODEL == 'ALEX' or MODEL == 'VGG':
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES, bias=True)
    )
elif MODEL == 'DENSE':
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES, bias=True)
    )
else:
    pass

# avoid training pretrained model
for layer in list(model.children())[:-LAYERS_TO_TRAIN]:
    for param in layer.parameters():
        param.requires_grad = False

print(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                       lr=LR,
#                       momentum=MOMENTUM,
#                       weight_decay=WEIGHT_DECAY)
optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

model.loss = nn.CrossEntropyLoss()
model.testLoss = nn.CrossEntropyLoss(reduction='sum')


class ImageLoader(object):

    def __init__(self, batchSize):
        super(ImageLoader, self).__init__()
        zoom = 1.15
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=input_size),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(1, 1.2), shear=0.1),
            transforms.RandomCrop(size=input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size=input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.ImageFolder(root='chest_xray_dataset/train',
                                             transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batchSize,
                                                       shuffle=True,
                                                       num_workers=8)

        test_dataset = datasets.ImageFolder(root='chest_xray_dataset/test',
                                            transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=batchSize,
                                                      shuffle=False,
                                                      num_workers=8)

        self.classes = ('normal', 'pneumonia')


loader = ImageLoader(batchSize=BATCH_SIZE)
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


if __name__ == '__main__':
    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = train(model, device, loader.trainloader, optimizer, epoch, LOG_INTERVAL)
            test_loss, test_accuracy = test(model, device, loader.testloader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            train_accuracies.append((epoch, train_accuracy))
            test_accuracies.append((epoch, test_accuracy))
    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        xs = [p[0] for p in train_losses]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(xs, [p[1] for p in train_accuracies], 'g-^', label='Train Accu')
        ax1.plot(xs, [p[1] for p in test_accuracies], 'g-o', label='Test Accu')
        ax2.plot(xs, [p[1] for p in train_losses], 'b-^', label='Train Loss')
        ax2.plot(xs, [p[1] for p in test_losses], 'b-o', label='Test Loss')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')

        plt.legend(handles=[mlines.Line2D([], [], color='black', marker='^', label='Train'),
                            mlines.Line2D([], [], color='black', marker='o', label='Test')])

        plt.savefig('{0}_{1}_{2}_result'.format(MODEL, EPOCHS, LR))
        with open('{0}_{1}_{2}_result'.format(MODEL, EPOCHS, LR) + '_data.dat', 'w') as f:
            f.write(str(train_accuracies) + '\n')
            f.write(str(test_accuracies) + '\n')
            f.write(str(train_losses) + '\n')
            f.write(str(test_losses) + '\n')


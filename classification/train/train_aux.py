#%% resnet
# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import ResNet, ResidualBlock


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

#%% mnist lenet
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import ConvNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
# Controlling source of randomness: pytorch RNG
torch.manual_seed(0)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

#%% cifar lenet
#%%
import matplotlib.pyplot as plt 
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

# controlling source of randomness: pytorch RNG
torch.manual_seed(0)

#%% 
# custom dataset, dataloader and transforms using pytorch and torchvision
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('{}'.format(classes[labels[j]]) for j in range(4)))

#%%
# define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%
# training loop
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#%% 
# test
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('{}'.format(classes[labels[j]]) for j in range(4)))

# load model
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('{}'.format(classes[labels[j]]) for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#%% 
# gpu training and testing
# need to set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
# move the model to GPU
net.to(device)
# move the data to gpu
inputs, labels = data[0].to(device), data[1].to(device)

#%% cifar truncated
from datasets import CIFAR10_truncated
import argparse
from torchvision import transforms
import torch.utils.data as data

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import ConvNet, CifarNet, ResNet, ResidualBlock
from losses import LDAMLoss, FocalLoss, SEQLLoss, SEQLLoss_beta

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyper parameters
num_epochs = 200
num_classes = 10
learning_rate = 0.001
batch_size = 256
TRAIN_BATCHSIZE = batch_size
TEST_BATCHSIZE = batch_size
# Controlling source of randomness: pytorch RNG
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="./data/cifar10", help="Data directory")
parser.add_argument('--train_bs', default=TRAIN_BATCHSIZE, type=int, help='training batch size')
parser.add_argument('--test_bs', default=TEST_BATCHSIZE, type=int, help='testing batch size')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
args = parser.parse_args()

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform, download=True)
    test_ds = CIFAR10_truncated(datadir, train=False, transform=transform, download=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)
    
    total_train = len(train_ds)
    total_test = len(test_ds)
    print(len(train_ds))
    print(len(test_ds))

    return train_dl, test_dl, total_train, total_test

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    dataidxs = []
    #load the index of imbalanced CIFAR-10 from dataidx.txt
    with open("dataidx.txt", "r") as f:
        for line in f:
            dataidxs.append(int(line.strip()))
    #get the training/testing data loader
    train_dl, test_dl, total_train, total_test = get_dataloader(args.datadir, args.train_bs, args.test_bs, dataidxs)
    
    class_total = list(0. for i in range(10))
    for _, labels in train_dl:
        for i in range(len(labels)):
                labels = labels.to(device)
                label = labels[i]
                class_total[label] += 1
    for i in range(10):
            print('Total objects in class no. {} ({}): {}.'.format(i, classes[i],
            class_total[i]))

    #model = ConvNet(num_classes).to(device)
    #model = CifarNet().to(device)
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

    # Loss
    per_cls_weights = None

    '''
    # reweighting of class scheme
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    '''

    #criterion = nn.CrossEntropyLoss()
    # not always necessary, if it has internal states or parameters then push to gpu
    #criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
    #criterion = LDAMLoss(cls_num_list=class_total, max_m=0.5, s=30, weight=per_cls_weights).to(device)       
    #criterion = FocalLoss(weight=per_cls_weights, gamma=1).to(device)

    freq_class = np.array(class_total)/total_train
    #criterion = SEQLLoss(freq_info=freq_class).to(device)
    criterion = SEQLLoss_beta(freq_info=freq_class).to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_dl)
    #total_train = total_step * TRAIN_BATCHSIZE # this assumes all batch sizes have 
    # same number of images but the last one wont
    curr_lr = learning_rate

    training_proc_avg = []
    test_proc_avg = []
    for epoch in range(num_epochs):
        current_losses = []
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                print ('Epoch [{}/{}], Samples [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, (i+1) * TRAIN_BATCHSIZE, total_train, loss.item()))
                current_losses.append(loss.item()) # appends the current value of the loss into a list
        
        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


        training_proc_avg.append(mean(current_losses)) # calculates mean of losses for current epoch and appends to list of avgs

        if (epoch) % 10 == 0: # each 10 epochs calculate the test set
            current_losses_test = []
            with torch.no_grad():
                correct = 0
                total = 0

                for images, labels in test_dl:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    current_losses_test.append(loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                test_loss_epoch_mean = mean(current_losses_test)
                print('Current loss and accuracy on test set at epoch no. {}: {}, {}%'.format(
                    epoch + 1, test_loss_epoch_mean, 100 * correct / total))        
                test_proc_avg.append(test_loss_epoch_mean)            


    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            current_losses_test.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        test_proc_avg.append(mean(current_losses_test))

        for i in range(10):
            print('Total objects in class no. {} ({}): {}. Accuracy: {}'.format(i, classes[i],
            class_total[i], 100 * class_correct[i] / class_total[i]))
    
    plots(training_proc_avg, test_proc_avg)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

def plots(training_proc_avg, test_proc_avg):
    x = np.arange(1, len(training_proc_avg)+1)
    x_2 = np.linspace(1, len(training_proc_avg)+1, len(test_proc_avg))

    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, training_proc_avg, label='Training loss')
    axs.plot(x_2, test_proc_avg, label='Testing loss')
    axs.set_xlabel('Epoch no.')
    axs.set_ylabel('Average loss for epoch')
    axs.set_title('Loss as training progresses')
    axs.legend()

    if not os.path.exists('results'):
        os.makedirs('results')
    fig.savefig('./results/training_loss.png', dpi=300)

if __name__ == '__main__':
    main()
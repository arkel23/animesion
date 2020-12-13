import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary

from utils.utils import plot_losses, show_results #, data_loading
from models.models import simpleNet
from data.datasets import moeImouto, danbooruFacesCrops

from pytorch_pretrained_vit import ViT

def update_lr(optimizer, lr): 
    # For updating learning rate   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def data_loading(args, train, transform=False):
    if args.dataset_name == 'moeImouto':
        dataset = moeImouto(input_size=args.image_size, train=train)
        classes = dataset.classes

    elif args.dataset_name == 'danbooruFacesCrops':
        pass
        '''
        # NEEDS UPDATING
        dataset = danbooruFacesCrops(split=split, transform=True)
        classes = dataset.classes
        dataset.stats()
        '''
    dataset_loader = data.DataLoader(dataset,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=4)

    '''
    # NEEDS UPDATING
    if visualization==True:
        classes_print(dataset)
        img_grid(classes, dataset_loader, batch_size=16)
    '''

    return dataset_loader, classes

def train(logger, args):

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # dataloader
    tfms = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # training
    train_loader, classes = data_loading(args, train=True)
    # validation
    # TO DO SEPARATE THE TRAINING INTO TRAIN AND VAL SET USING CROSS-VALIDATION
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    # testing
    test_loader, classes = data_loading(args, train=False)
    no_classes = len(classes)

    # initiates model and loss     
    if args.model_type=='shallow':
        model = simpleNet(no_classes)
    elif args.model_type=='resnet18':
        model = torchvision.models.resnet18(pretrained=args.pretrained)
    elif args.model_type=='resnet152':
        model = torchvision.models.resnet152(pretrained=args.pretrained)
    model.to(device)
    # TO DO: FINISH ADDING THE TRNSFORMER MODELS
    '''
    model_list = ['B_16', 'B_32', 'L_32', 'B_16_imagenet1k',
    'B_32_imagenet1k', 'L_16_imagenet1k', 'L_32_imagenet1k']
    #model_name = model_list[0]
    #model = ViT(model_name, pretrained=True, num_classes=no_classes, image_size=args.image_size)
    #model.to(device)
    '''

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # prints model summary (layers, parameters by giving it a sample input)
    #summary(model, input_size=iter(train_loader).next()[0].shape[1:])
    
    # Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    training_proc_avg = []
    test_proc_avg = []

    for epoch in range(args.train_epochs):
        current_losses = []
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

            if (i % args.checkpoint_each_epochs) == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}"
                    .format(epoch+1, args.train_epochs, i+1, total_step, loss.item()))
                # appends the current value of the loss into a list
                current_losses.append(loss.item()) 
                
        # Decay learning rate
        if (epoch+1) % args.epoch_decay == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
            
        # calculates mean of losses for current epoch and appends to list of avgs
        training_proc_avg.append(mean(current_losses)) 

        # validates on test set once per epoch (or a few no of epochs)
        # TO DO: ADD EARLY STOPPING IF TEST_PROC_AVG[current]>test_proc[prev] break;
        validate(device=device, batch_size=args.batch_size, 
        classes=classes,
        model=model, criterion=criterion, no_classes=no_classes, 
        training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
        loader=test_loader, dataset_name=args.dataset_name, last=False)

    # validate on test set and plot results
    validate(device=device, batch_size=args.batch_size, 
    classes=classes,
    model=model, criterion=criterion, no_classes=no_classes, 
    training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
    loader=test_loader, dataset_name=args.dataset_name, last=True)

    # Save the model checkpoint
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.ckpt'))

    logger.info('Finished training successfully.')
    
def validate(device, batch_size, classes,
model, criterion, no_classes, 
training_proc_avg, test_proc_avg, loader, dataset_name=False, last=False):
    # Test the model (validation set)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        current_losses_test = []

        class_correct = list(0. for i in range(no_classes))
        class_total = list(0. for i in range(no_classes))
        
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
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

        print('Test Accuracy of the model on the test images: {:.4f} %'.format(100 * correct / total))
        test_proc_avg.append(mean(current_losses_test))

        if last==True:
            
            # plot loss
            plot_losses(training_proc_avg, test_proc_avg)

            '''
            # NEEDS UPDATING for new modularity and program structure
            if dataset_name == 'moeImouto':
                for i in range(no_classes):
                    print('Total objects in class no. {} ({}): {:d}. Accuracy: {:.4f}'.format(i, classes[i],
                    int(class_total[i]), 100 * class_correct[i] / class_total[i]))

            # show examples of classified images
            show_results(device, loader, model, classes)
            '''

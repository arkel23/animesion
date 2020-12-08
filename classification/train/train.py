import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import torch
import torch.nn as nn
from torchsummary import summary

from utils.utils import update_lr, plot_losses, data_loading, show_results
from models.models import simpleNet

def train():
    # trains model from scratch
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # train constants
    no_epochs = 10
    save_iter = 10
    epoch_decay = 100 
    batch_size = 256
    learning_rate = 0.001

    # dataloader
    # training
    train_loader, classes = data_loading(batch_size, visualization=True)
    no_classes = len(classes)
    # validation
    # testing

    # initiates model and loss     
    model = simpleNet(no_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # prints model summary (layers, parameters by giving it a sample input)
    summary(model, input_size=iter(train_loader).next()[0].shape[1:])
    
    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    training_proc_avg = []
    test_proc_avg = []

    for epoch in range(no_epochs):
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
            
            if (i+1) % save_iter == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}"
                    .format(epoch+1, no_epochs, i+1, total_step, loss.item()))
                # appends the current value of the loss into a list
                current_losses.append(loss.item()) 
                
        # Decay learning rate
        if (epoch+1) % epoch_decay == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
            
        # calculates mean of losses for current epoch and appends to list of avgs
        training_proc_avg.append(mean(current_losses)) 

        # validates on test set once per epoch (or a few no of epochs)
        # TO DO: ADD EARLY STOPPING IF TEST_PROC_AVG[current]>test_proc[prev] break;
        validate(device=device, batch_size=batch_size, 
        classes=classes,
        model=model, criterion=criterion, no_classes=no_classes, 
        training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
        loader=train_loader, last=False)

    # validate on test set and plot results
    validate(device=device, batch_size=batch_size, 
    classes=classes,
    model=model, criterion=criterion, no_classes=no_classes, 
    training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
    loader=train_loader, last=True)

    # Save the model checkpoint
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.ckpt'))
    
def validate(device, batch_size, classes,
model, criterion, no_classes, 
training_proc_avg, test_proc_avg, loader, last=False):
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
            for i in range(no_classes):
                print('Total objects in class no. {} ({}): {:d}. Accuracy: {:.4f}'.format(i, classes[i],
                int(class_total[i]), 100 * class_correct[i] / class_total[i]))

            # plot loss
            plot_losses(training_proc_avg, test_proc_avg)

            # show examples of classified images
            show_results(device, loader, model, classes)
        

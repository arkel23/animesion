import os
import argparse

import torch

from train import data_loading, model_selection, validate

# take main from train.py and modify it for inference (choose model and checkpoint or not)
# input an image (or a dataset)
# if image then calculate the class and output results/labels
# if dataset then output the img grid along with the ground truth and predicted labels
# also output the testing accuracy for the desired dataset along with the 
# per label accuracy

# PER CLASS_ACCURACY
# PREVIOUSLY IN VALIDATE AFTER CALCULATING TOP-K ACCURACY

def show_results(device, loader, model, classes, batch_size=8):
    images, labels = iter(loader).next()[:batch_size]
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    print('\n'.join('Correct: {}, Predicted: {}'.format(
    classes[labels[j]], classes[predicted[j]]) for j in range(batch_size)))
    bottom_text = '\n'.join('Correct: {}, Predicted: {}'.format(
    classes[labels[j]], classes[predicted[j]]) for j in range(batch_size))
    imshow(torchvision.utils.make_grid(images.cpu()[:batch_size]), 'class_results', bottom_text)

def test_set():
    # validate on test set and plot results
    validate(device=device, model=model, criterion=criterion,
    no_classes=no_classes, loader=test_loader, 
    top1_accuracies=top1_accuracies, top5_accuracies=top5_accuracies,
    classid_classname_dic=classid_classname_dic)


def validate(device, model, criterion, no_classes, loader,
    top1_accuracies, top5_accuracies, 
    classid_classname_dic, val_loss_avg=[]):
    # Test the model (validation set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout probability goes to 0
    model.eval()
    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        total = 0
        current_losses = []

        class_correct = list(0. for i in range(no_classes))
        class_total = list(0. for i in range(no_classes))
        
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)    
            current_losses.append(loss.item())
            
            # calculate top-k (1 and 5) accuracy
            total += labels.size(0)
            curr_corr_list = accuracy(outputs.data, labels, (1, 5, ))
            correct_1 += curr_corr_list[0]
            correct_5 += curr_corr_list[1]    
            
            # calculate per-class accuracy
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        # append avg val loss
        val_loss_avg.append(mean(current_losses))

        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1/total
        top1_accuracies.append(curr_top1_acc)
        print('Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %'.format(curr_top1_acc))
        curr_top5_acc = 100 * correct_5/total
        top5_accuracies.append(curr_top5_acc)
        print('Val/Test Top-5 Accuracy of the model on the test images: {:.4f} %'.format(curr_top5_acc))

        # compute per class accuracy
        for i in range(no_classes):
            class_accuracy = 100 * class_correct[i] / class_total[i]
            class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==i, 'class_name'].item()
            print('Total objects in class no. {} ({}): {}. Accuracy: {}'.format(
            i, class_name, class_total[i], class_accuracy))
    

        return curr_top1_acc


the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))


'''
# NEEDS UPDATING
if visualization==True:
    classes_print(dataset)
    img_grid(classes, dataset_loader, batch_size=16)
'''

'''
        if last==True:
            pass
            # plot loss
            #plot_losses(training_proc_avg, test_proc_avg)

            # NEEDS UPDATING for new modularity and program structure
            if dataset_name == 'moeImouto':
                for i in range(no_classes):
                    print('Total objects in class no. {} ({}): {:d}. Accuracy: {:.4f}'.format(i, classes[i],
                    int(class_total[i]), 100 * class_correct[i] / class_total[i]))

            # show examples of classified images
            show_results(device, loader, model, classes)
'''

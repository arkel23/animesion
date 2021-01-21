import os
import logging
import argparse
import pandas as pd
from statistics import mean
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsummary import summary

from train import data_loading, model_selection

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
def inference(args):
    # makes results_dir if doesn't exist
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # General dataset
    data_set, _ = data_loading(args, split='test', inference=True)
    no_classes = data_set.no_classes
    classid_classname_dic = data_set.classes

    # Image to be tested
    transform = data_set.transform
    image = Image.open(args.image_path)
    image_transformed = transform(image)
    image_batch = torch.unsqueeze(image_transformed, 0).to(device)
    print(image.size, image_transformed.shape, image_batch.shape)

    # model
    model = model_selection(args, no_classes)
    model.to(device)    
    model.load_state_dict(torch.load(args.checkpoint_path))
    # prints model summary (layers, parameters by giving it a sample input)
    if args.vis_arch:
        summary(model, input_size=image_batch.shape[1:])

    # don't calculate gradients and put model into evaluation mode (no dropout/batch norm/etc)
    model.eval()
    with torch.no_grad():
        outputs = model(image_batch).squeeze(0)
        for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
            prob = torch.softmax(outputs, -1)[idx].item() * 100
            class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
            print('Prediction No. {}: {} [ID: {}], Confidence: {}'.format(i+1, class_name, idx, prob))

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces"], 
                        required=True, help="Which dataset to use (for no. of classes/loading model).")
    parser.add_argument("--dataset_path", required=True,
                        help="Path for the dataset.")
    parser.add_argument("--image_path", required=True,
                        help="Path for the image to be tested.")
    parser.add_argument("--image_size", choices=[128, 224], default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'],
                        required=True,
                        help="Which model architecture to use")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path for model checkpoint to load.")    
    parser.add_argument("--results_dir", default="results_inference", type=str,
                        help="The directory where results will be stored.")
    parser.add_argument("--pretrained", type=bool, default=True,
                        help="DON'T CHANGE! Always true since always loading when doing inference.")
    parser.add_argument("--vis_arch", type=bool, default=False,
                        help="Visualize architecture through model summary.")
               
    args = parser.parse_args()

    inference(args)            

if __name__ == '__main__':
    main()
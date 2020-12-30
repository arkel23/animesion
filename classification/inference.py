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

'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
'''

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

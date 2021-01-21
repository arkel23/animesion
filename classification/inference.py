import os
import logging
import argparse
<<<<<<< HEAD
import pandas as pd
from statistics import mean
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsummary import summary
=======
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torchvision
import torchvision.transforms as transforms
>>>>>>> 092563651ee2caee8ef7ef9ac44eeea58e4555a2

from train import data_loading, model_selection

# take main from train.py and modify it for inference (choose model and checkpoint or not)
# input an image (or a dataset)
# if image then calculate the class and output results/labels
# if dataset then output the img grid along with the ground truth and predicted labels
# also output the testing accuracy for the desired dataset along with the 
# per label accuracy

# PER CLASS_ACCURACY
# PREVIOUSLY IN VALIDATE AFTER CALCULATING TOP-K ACCURACY
 
def imshow(inp, out_name, title=None, imagenet_values=False, save_results=False):
    '''Imshow for Tensor.
    # pretrained on imagenet (resnets)
    # std=(0.229, 0.224, 0.225)
    # mean=(0.485, 0.456, 0.406)

    # others:
    # std=(0.5, 0.5, 0.5)
    # mean=(0.5, 0.5, 0.5)
    '''
    if imagenet_values:
        inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255])
    else:
        inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5])

    inv_tensor = inv_normalize(inp)
    inp = inv_tensor.to('cpu').numpy().transpose((1, 2, 0))
    inp = np.uint8(np.clip(inp, 0, 1) * 255)

    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=10, wrap=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    if save_results:
        plt.savefig('{}'.format(out_name), dpi=300)
    plt.close()

def inference(args, device, model, data_set):
    classid_classname_dic = data_set.classes
    transform = data_set.transform
    
    # Images to be tested
    file_list = [os.path.join(args.images_path, f) for f in os.listdir(args.images_path) if os.path.isfile(
        os.path.join(args.images_path, f))]
    
    # don't calculate gradients and put model into evaluation mode (no dropout/batch norm/etc)
    model.eval()
    with torch.no_grad():
        for image_dir in file_list:
            # read image one by one and apply transforms
            file_name_no_ext = os.path.splitext(os.path.split(image_dir)[1])[0]
            out_name = os.path.join(args.results_dir, '{}.jpg'.format(file_name_no_ext))
            image = Image.open(image_dir)
            if image.mode != 'RGB':
                print("Image {} should be RGB".format(image_dir))
                continue
            image_transformed = torch.unsqueeze(transform(image), 0).to(device)
            print('File: {}, Original image size: {}, Size after reshaping and unsqueezing: {}'.format(
                image_dir, image.size, image_transformed.shape))

            # calculate outputs for each image
            outputs = model(image_transformed).squeeze(0)
            classes_predicted = []
            classes_predicted.append(file_name_no_ext)
            classes_predicted.append('\n')
            for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                prob = torch.softmax(outputs, -1)[idx].item() * 100
                class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                classes_predicted.append(predict_text)
                print(predict_text, end='')

            classes_predicted = '  '.join(classes_predicted)
            grid = torchvision.utils.make_grid(image_transformed)
            imshow(grid, out_name, title=classes_predicted, save_results=args.save_results)

def environment_loader(args):
    # makes results_dir if doesn't exist
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)
    
<<<<<<< HEAD

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
=======
    # General dataset
    data_set, data_loader = data_loading(args, split='test')
    no_classes = data_set.no_classes
>>>>>>> 092563651ee2caee8ef7ef9ac44eeea58e4555a2

    # model
    model = model_selection(args, no_classes)
    model.to(device)    
    model.load_state_dict(torch.load(args.checkpoint_path))
<<<<<<< HEAD
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
=======

    return device, model, data_set, data_loader
>>>>>>> 092563651ee2caee8ef7ef9ac44eeea58e4555a2

def main():
  
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
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
=======
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces"], default='danbooruFaces',
                        help="Which dataset to use (for no. of classes/loading model).")
    parser.add_argument("--dataset_path", default="data/danbooruFaces/",
                        help="Path for the dataset.")
    parser.add_argument("--images_path", default='test_images',
                        help="Path for the images to be tested.")
    parser.add_argument("--image_size", choices=[128, 224], default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'], default='L_32',
                        help="Which model architecture to use")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="checkpoints/danbooruFaces_l32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.ckpt",
>>>>>>> 092563651ee2caee8ef7ef9ac44eeea58e4555a2
                        help="Path for model checkpoint to load.")    
    parser.add_argument("--results_dir", default="results_inference", type=str,
                        help="The directory where results will be stored.")
    parser.add_argument("--pretrained", type=bool, default=True,
                        help="DON'T CHANGE! Always true since always loading when doing inference.")
<<<<<<< HEAD
    parser.add_argument("--vis_arch", type=bool, default=False,
                        help="Visualize architecture through model summary.")
               
    args = parser.parse_args()

    inference(args)            
=======
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for train/val/test. Just for loading the dataset.")
    parser.add_argument("--save_results", type=bool, default=True,
                        help="Save the images after transform and with label results.")
               
    args = parser.parse_args()

    device, model, data_set, data_loader = environment_loader(args) 

    inference(args, device, model, data_set)           
>>>>>>> 092563651ee2caee8ef7ef9ac44eeea58e4555a2

if __name__ == '__main__':
    main()
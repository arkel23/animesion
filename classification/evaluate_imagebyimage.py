import os
import logging
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torchvision
import torchvision.transforms as transforms

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
    inp = input('Press anything to keep visualizing: ')
    #plt.pause(5)
    #if save_results:
    #    plt.savefig('{}'.format(out_name), dpi=300)
    plt.close()

def evaluate_imagebyimage(args, device, model, data_set, data_loader):
    
    # related to dataset
    num_classes = data_set.num_classes
    classid_classname_dic = data_set.classes
    total_step = len(data_loader)
    curr_line = 'Total no. of samples in test set: {}\nTotal no. of classes: {}\n'.format(
        len(data_set), num_classes)
    print(curr_line)
    
    # don't calculate gradients and put model into evaluation mode (no dropout/batch norm/etc)
    model.eval()
    with torch.no_grad():
        #for image_dir in file_list:
        for i, (images, labels) in enumerate(data_loader):
            # prints current evaluation step after each 10 steps
            if (i % 10) == 0:
                print ("Step [{}/{}]".format(i+1, total_step))
            images = images.to(device)
            labels = labels.to(device)

            for j in range(len(images)):

                outputs = model(images[j, :].unsqueeze(0)).squeeze(0)
                label_class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==labels[j].item(), 'class_name'].item()
                
                predict_top1 = torch.topk(outputs, k=1).indices
                if (predict_top1 == labels[j].item()):
                    continue
                
                print('Ground truth label: ', label_class_name)
                
                classes_predicted = []
                classes_predicted.append('{}'.format(label_class_name))
                classes_predicted.append('\n')
                
                for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                    prob = torch.softmax(outputs, -1)[idx].item() * 100
                    class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                    predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                    classes_predicted.append(predict_text)
                    print(predict_text, end='')

                classes_predicted = '  '.join(classes_predicted)
                grid = torchvision.utils.make_grid(images[j, :])
                imshow(grid, out_name='placeholder', title=classes_predicted, save_results=args.save_results)

def environment_loader(args):
    # makes results_dir if doesn't exist
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Device configuration
    if hasattr(args, 'vis_attention'):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)
    
    # General dataset
    data_set, data_loader = data_loading(args, split='test')
    args.num_classes = data_set.num_classes

    # model
    model = model_selection(args)
    model.to(device)    
    model.load_state_dict(torch.load(args.checkpoint_path))

    return device, model, data_set, data_loader

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces", "danbooruFull"], default='danbooruFaces',
                        help="Which dataset to use (for no. of classes/loading model).")
    parser.add_argument("--dataset_path", 
                        default="/hdd/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/",
                        help="Path for the dataset.")
    parser.add_argument("--images_path", default='test_images',
                        help="Path for the images to be tested.")
    parser.add_argument("--image_size", choices=[128, 224], default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--model_name", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'], default='L_16',
                        help="Which model architecture to use")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="./checkpoints/verify_danbooruFaces_l16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt",
                        help="Path for model checkpoint to load.")    
    parser.add_argument("--results_dir", default="results_inference", type=str,
                        help="The directory where results will be stored.")
    parser.add_argument("--pretrained", type=bool, default=True,
                        help="DON'T CHANGE! Always true since always loading when doing inference.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for train/val/test. Just for loading the dataset.")
    parser.add_argument("--save_results", type=bool, default=False,
                        help="Save the images after transform and with label results.")  
                               
    args = parser.parse_args()
    args.load_partial_mode = None
    args.transfer_learning = False
    print(args)

    device, model, data_set, data_loader = environment_loader(args)

    evaluate_imagebyimage(args, device, model, data_set, data_loader)           

if __name__ == '__main__':
    main()

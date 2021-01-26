import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import cv2

import torch
import torchvision
import torchvision.transforms as transforms

from pytorch_pretrained_vit import ViT

from inference import environment_loader

def inference_attention(args, device, model, data_set):
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
            #outputs = model(image_transformed).squeeze(0)
            outputs, att_mat = model(image_transformed)
            outputs = outputs.squeeze(0)
            vis_attention(args, image, outputs, att_mat, file_name_no_ext)

            
            classes_predicted = []
            classes_predicted.append(file_name_no_ext)
            classes_predicted.append('\n')
            for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                prob = torch.softmax(outputs, -1)[idx].item() * 100
                class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                classes_predicted.append(predict_text)
                print(predict_text, end='')
            '''
            classes_predicted = '  '.join(classes_predicted)
            grid = torchvision.utils.make_grid(image_transformed)
            imshow(grid, out_name, title=classes_predicted, save_results=args.save_results)
            '''

def vis_attention(args, image, outputs, att_mat, file_name_no_ext):

    #outputs = outputs.squeeze(0)
    #print(outputs.shape)
    #print(len(att_mat))
    #print(att_mat[0].shape)
    #print(outputs, att_mat)
    #print('logits_size and att_mat sizes: ', outputs.shape, att_mat.shape)

    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.shape)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    #print(att_mat.shape)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print('residual_att and aug_att_mat sizes: ', residual_att.shape, aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1] # last layer output attention map
    #print('joint_attentions and last layer (v) sizes: ', joint_attentions.shape, v.shape)
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #print(mask.shape)
    mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
    #print(mask.shape)
    result = (mask * image).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(image)
    _ = ax2.imshow(result)

    print('-----')
    if not os.path.exists(os.path.join(args.results_dir, 'attention')):
        os.mkdir(os.path.join(args.results_dir, 'attention'))

    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        #print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

    i = 0
    v = 0
    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
        result = (mask * image).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        title = 'AttentionMap_Layer{}'.format(i+1)
        ax2.set_title(title)
        _ = ax1.imshow(image)
        _ = ax2.imshow(result)
        out_name = '{}_{}.jpg'.format(file_name_no_ext, title)
        plt.savefig(os.path.join(args.results_dir, 'attention', out_name))
        plt.close()
    i = 0
    v = 0
    


def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces"], default='danbooruFaces',
                        help="Which dataset to use (for no. of classes/loading model).")
    parser.add_argument("--dataset_path", default="data/danbooruFaces/",
                        help="Path for the dataset.")
    parser.add_argument("--images_path", default='test_images',
                        help="Path for the images to be tested.")
    parser.add_argument("--image_size", choices=[128, 224], default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'], default='B_32',
                        help="Which model architecture to use")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="checkpoints/danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.ckpt",
                        help="Path for model checkpoint to load.")    
    parser.add_argument("--results_dir", default="results_inference", type=str,
                        help="The directory where results will be stored.")
    parser.add_argument("--pretrained", type=bool, default=True,
                        help="DON'T CHANGE! Always true since always loading when doing inference.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for train/val/test. Just for loading the dataset.")
    parser.add_argument("--save_results", type=bool, default=True,
                        help="Save the images after transform and with label results.")
    parser.add_argument("--vis_attention", type=bool, default=True,
                        help="Saves attention maps.")
               
    args = parser.parse_args()

    device, model, data_set, data_loader = environment_loader(args) 

    inference_attention(args, device, model, data_set)           

if __name__ == '__main__':
    main()
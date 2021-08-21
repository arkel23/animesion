import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from train import environment_loader
import utilities as utilities
from utilities.build_vocab import Vocabulary

DEFAULT_MAX_TEXT_SEQ_LEN = 16

def imshow(inp, out_name, title=None, save_results=False):
    
    inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5])

    inv_tensor = inv_normalize(inp)
    inp = inv_tensor.to('cpu').numpy().transpose((1, 2, 0))
    inp = np.uint8(np.clip(inp, 0, 1) * 255)

    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=8, wrap=True)
    plt.tight_layout()
    plt.show(block=False)
    inp = input("Input 'exit'  to stop visualizing: ")
    if inp == 'exit':
        sys.exit()
    #plt.pause(5)
    #if save_results:
    #    plt.savefig('{}'.format(out_name), dpi=300)
    plt.close()
    

def evaluate(args, device, model, data_set, data_loader, mask_scheduler, tokenizer):
    
    og_file_name = '{}'.format(os.path.splitext(os.path.split(args.checkpoint_path)[1])[0])
    log_file_name = 'evaluate_{}.txt'.format(og_file_name)
    f = open(os.path.join(args.results_dir, '{}'.format(log_file_name)), 'w')

    # related to dataset
    num_classes = data_set.num_classes
    classid_classname_dic = data_set.classes
    total_steps = len(data_loader)
    curr_line = 'Total no. of samples in test set: {}\nTotal no. of classes: {}\n'.format(
        args.batch_size*total_steps, num_classes)
    utilities.misc.print_write(f, curr_line)

    # Test the model (test set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout probability goes to 0
    model.eval()
    if args.vis_arch:
        summary(model, input_size=iter(data_loader).next()[0].shape[1:])
    
    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        total = 0

        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        
        for i, batch in enumerate(data_loader):
            if args.multimodal:
                images, labels, captions = batch
                captions = captions.squeeze(dim=1).to(device)
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
                    
            # return new masks and updated caption tokens according to schedule
            if mask_scheduler is not None:
                captions_updated, labels_text = mask_scheduler.ret_mask(total_steps, captions)
            else:
                labels_text = None
            
            # Forward pass
            if args.multimodal and args.mask_schedule and args.exclusion_loss:
                outputs, outputs_text, exclusion_loss = model(images, text=captions_updated)
            elif args.multimodal and args.mask_schedule:
                outputs, outputs_text = model(images, text=captions_updated)
            elif args.multimodal and args.exclusion_loss:
                outputs, exclusion_loss = model(images, text=captions)
            elif args.multimodal:
                outputs = model(images, text=captions)
            elif args.exclusion_loss:
                outputs, exclusion_loss = model(images)
            else:
                outputs = model(images)
                        
            # calculate top-k (1 and 5) accuracy
            total += labels.size(0)
            curr_corr_list = utilities.misc.accuracy(outputs.data, labels, (1, 5, ))
            correct_1 += curr_corr_list[0]
            correct_5 += curr_corr_list[1]

            # calculate per-class accuracy
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1

            if i % args.log_freq == 0:
                curr_line = "Validation/Test Step [{}/{}]\n".format(i+1, total_steps)
                utilities.misc.print_write(f, curr_line)

                if labels_text is not None:
                    utilities.misc.decode_text(f, tokenizer, outputs_text, captions, captions_updated, labels_text)

            #if save_all_captions and (mask_scheduler is not None):
            #    utilities.misc.decode_text(save_all_captions_file, tokenizer, outputs_text, captions, 
            #   captions_updated, labels_text, num_print=outputs_text.shape[0], save_all_captions=True)
            
        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1/total
        curr_line = 'Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %'.format(curr_top1_acc)
        utilities.misc.print_write(f, curr_line)
        
        curr_top5_acc = 100 * correct_5/total
        curr_line = 'Val/Test Top-5 Accuracy of the model on the test images: {:.4f} %'.format(curr_top5_acc)
        utilities.misc.print_write(f, curr_line)

        # compute per class accuracy
        class_id_name_nosamples_classacc_dic = {}

        for i in range(num_classes):
            class_accuracy = 100 * class_correct[i] / class_total[i]
            class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==i, 'class_name'].item()
            curr_line = 'Total objects in class no. {} ({}): {}. Top-1 Accuracy: {}\n'.format(
            i, class_name, class_total[i], class_accuracy)
            utilities.misc.print_write(f, curr_line)

            class_id_name_nosamples_classacc_dic[i] = [class_name, class_total[i], class_accuracy]
    
    df = pd.DataFrame.from_dict(class_id_name_nosamples_classacc_dic, orient='index', 
    columns=['class_name', 'num_samples', 'class_accuracy'])
    df['id'] = df.index
    print(df.head())
    df.sort_values(by=['num_samples', 'class_accuracy'], inplace=True, ascending=False)
    print(df.head())
    classes_results_file_name = 'classes_results_{}.csv'.format(og_file_name)
    df.to_csv(os.path.join(args.results_dir, '{}'.format(classes_results_file_name)), index=False, header=True)
    
    f.close()


def evaluate_imagebyimage(args, device, model, tokenizer, data_set, data_loader):
    
    # related to dataset
    num_classes = data_set.num_classes
    classid_classname_dic = data_set.classes
    total_steps = len(data_loader)
    curr_line = 'Total no. of samples in test set: {}\nTotal no. of classes: {}\n'.format(
        args.batch_size*total_steps, num_classes)
    print(curr_line)
    
    # don't calculate gradients and put model into evaluation mode (no dropout/batch norm/etc)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if args.multimodal:
                images, labels, captions = batch
                captions = captions.squeeze(dim=1).to(device)
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            if (i % args.log_freq) == 0:
                print ("Step [{}/{}]".format(i+1, total_steps))

            for j in range(len(images)):
                # Forward pass
                if args.multimodal:
                    caption = captions[j, :].unsqueeze(0)
                    outputs = model(images[j, :].unsqueeze(0), text=caption)
                else:
                    outputs = model(images[j, :].unsqueeze(0))
                outputs = outputs.squeeze(0)

                label_class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==labels[j].item(), 'class_name'].item()
                
                predict_top1 = torch.topk(outputs, k=1).indices
                if (predict_top1 == labels[j].item()):
                    continue
                
                if tokenizer is not None:
                    print('Ground truth tags: ', tokenizer.decode(caption.squeeze().cpu()))
                classes_predicted = []
                classes_predicted.append('Ground truth label: {}\n'.format(label_class_name))
                
                for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                    prob = torch.softmax(outputs, -1)[idx].item() * 100
                    class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                    predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                    classes_predicted.append(predict_text)
                    
                classes_predicted = '  '.join(classes_predicted)
                print(classes_predicted)
                grid = torchvision.utils.make_grid(images[j, :])
                imshow(grid, out_name='placeholder', title=classes_predicted, save_results=args.save_results)


def main():
  
    parent_parser = utilities.misc.ret_args(ret_parser=True)

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--vis_arch", action='store_true',
                        help="Visualize architecture through model summary.")
    parser.add_argument("--eval_imagebyimage", action='store_true',
                        help="Evaluate all or image by image")
    parser.add_argument("--save_results", action='store_true',
                        help="Save the images after transform and with label results.")   
    parser.set_defaults(results_dir='results_inference', no_epochs=1)
    args = parser.parse_args()
    
    (device, train_set, train_loader, val_loader, test_loader,
    classid_classname_dic, model, optimizer, lr_scheduler,
    mask_scheduler, tokenizer) = environment_loader(args, init=False)
    #print(args, model.configuration)

    if not args.eval_imagebyimage:
        evaluate(args, device, model, train_set, test_loader, mask_scheduler, tokenizer)
    else:
        assert args.mask_schedule==None, "Can't evaluate image by image if masking"
        evaluate_imagebyimage(args, device, model, tokenizer, train_set, test_loader)          

if __name__ == '__main__':
    main()

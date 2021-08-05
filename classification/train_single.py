import time
from datetime import timedelta
import os
import logging
import argparse
import pandas as pd

import torch
from torchsummary import summary
import wandb

import models.models as models
import data.datasets as datasets
import utils.utilities as utilities
from engine import train_one_epoch, validate

logger = logging.getLogger(__name__)

def train_main(logger, args):

    # Init timer and logger    
    time_start = time.time()
    wandb.init(config=args)
    wandb.run.name = '{}'.format(args.run_name)
    file_name = '{}_log.txt'.format(args.run_name)
    f = open(os.path.join(args.results_dir, '{}'.format(file_name)), 'w', buffering=1)
    utilities.print_write(f, str(args))
    # Set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utilities.set_seed(args.seed)

    # dataloader and train/test datasets
    train_set, train_loader = datasets.data_loading(args, split='train')
    _, val_loader = datasets.data_loading(args, split='val')
    _, test_loader = datasets.data_loading(args, split='test')
    args.num_classes = train_set.num_classes
    classid_classname_dic = train_set.classes

    # model
    model = models.model_selection(args, device)
    utilities.print_write(f, str(model.configuration))
    if not args.interm_features_fc:
        summary(model, input_size=iter(train_loader).next()[0].shape[1:])
    
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)

    steps_per_epoch = len(train_loader)
    steps_total = args.no_epochs * steps_per_epoch
    if args.lr_scheduler == 'warmupCosine':
        lr_scheduler = utilities.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=steps_total)
    else:
        lr_scheduler=None
    
    # Train the model
    train_loss_avg = []
    val_loss_avg = []
    top1_accuracies = []
    top5_accuracies = []
    best_epoch = 0
    curr_acc = 0
    top_acc = 0
    max_memory = 0
    
    for epoch in range(args.no_epochs):
        time_start_epoch = time.time()

        train_one_epoch(args, f, epoch, model, device,
        criterion, optimizer, lr_scheduler, train_loader, train_loss_avg)

        curr_max_memory = torch.cuda.max_memory_reserved()/(1024**3)
        if max_memory < curr_max_memory:
            max_memory = curr_max_memory
        
        # validates on test set once per epoch, calculates top1/5 acc and val loss avg
        curr_acc = validate(args, f, device=device, model=model, criterion=criterion, loader=val_loader,
        top1_accuracies=top1_accuracies, top5_accuracies=top5_accuracies, val_loss_avg=val_loss_avg)

        # Save the model checkpoint if the top1-acc is higher than current highest
        if curr_acc > top_acc:
            torch.save(model.state_dict(), os.path.join(args.results_dir, 
            '{}.ckpt'.format(args.run_name)))
            top_acc = curr_acc
            best_epoch = epoch + 1
        
    # validate on test set and plot results
    validate(args, f, device=device, model=model, criterion=criterion, loader=test_loader, 
    top1_accuracies=top1_accuracies, top5_accuracies=top5_accuracies)

    time_end = time.time()
    time_all = time_end - time_start

    curr_line = '''\n{}
    \nAbove configuration finished training successfully. 
    \nBest val accuracy: {}, at epoch no: {}/{}.
    \nHighest reserved memory: {} (GB).
    \nTotal time (loading, training and evaluation): {} seconds. Average: {} seconds.
    \nTime to reach top accuracy: {} seconds.
    \n'''.format(str(args),
    top_acc, best_epoch, args.no_epochs, 
    max_memory, 
    time_all, time_all/args.no_epochs,
    best_epoch * (time_all/args.no_epochs))
    utilities.print_write(f, curr_line)
    logger.info(curr_line)

    # contains the top1/5 accuracies for the validation after each epoch, and the last one for the test
    df_accuracies = pd.DataFrame(list(zip(top1_accuracies, top5_accuracies)))
    # contains the training and validation loss averages for each epoch
    df_losses = pd.DataFrame(list(zip(train_loss_avg, val_loss_avg)))

    df_metrics = pd.concat([df_accuracies, df_losses], axis=1)
    df_metrics.columns = ['top1_acc', 'top5_acc', 'train_loss_avg', 'val_loss_avg']
    df_metrics.to_csv(os.path.join(args.results_dir, 
    '{}_metrics.csv'.format(args.run_name)), sep=',', header=True, index=False)

    wandb.run.summary['Best top-1 accuracy'] = top_acc
    wandb.run.summary['Best epoch'] = best_epoch
    wandb.run.summary['Time total (s)'] = time_all
    wandb.run.summary['Average time per epoch (s)'] = time_all/args.no_epochs
    wandb.run.summary['Time to reach top accuracy'] = best_epoch * (time_all/args.no_epochs)
    wandb.run.summary['Peak memory consumption (GB)'] = max_memory

    wandb.finish()

def main():

    logging.basicConfig(filename='logs.txt', level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['moeImouto', 'danbooruFaces', 'cartoonFace', 'danbooruFull'], 
                        default='moeImouto', help='Which dataset to use.')
    parser.add_argument('--dataset_path', required=True, help='Path for the dataset.')
    parser.add_argument('--model_name', choices=['shallow', 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32'], default='shallow',
                        help='Which model architecture to use')
    parser.add_argument('--results_dir', default='results_training', type=str,
                        help='The directory where results will be stored')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image (square) resolution size')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for train/val/test.')
    parser.add_argument('--no_epochs', default=200, type=int,
                        help='Total number of epochs for training.')                         
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate.')  
    parser.add_argument('--lr_scheduler', type=str, choices=['warmupCosine', 'epochDecayConstant'], 
                        default='epochDecayConstant', help='LR scheduler.')
    parser.add_argument('--epoch_decay', default=50, type=int,
                        help='After how many epochs to decay the learning rate once.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR scheduler.')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='For models with pretrained weights available'
                        'Default=False')
    parser.add_argument('--checkpoint_path', type=str, 
                        default=None)     
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='Load partial state dict for transfer learning'
                        'Resets the [embeddings, logits and] fc layer for ViT'
                        'Resets the fc layer for Resnets'
                        'Default=False')    
    parser.add_argument('--load_partial_mode', choices=['full_tokenizer', 'patchprojection', 'posembeddings', 'clstoken', 
        'patchandposembeddings', 'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                        help='Load pre-processing components to speed up training')
    parser.add_argument('--log_freq', default=100, type=int,
                        help='Frequency in steps to print results (and save images if needed).')        
    parser.add_argument('--no_cpu_workers', type=int, default=8, help='CPU workers for data loading.')
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--interm_features_fc', type=bool, default=False, 
                        help='Create FC using intermediate features instead of only last layer.')
    parser.add_argument('--debugging', type=bool, default=False,
                        help='If true then shortens the training/val loops to log_freq*3.')
    parser.add_argument('--exclusion_loss', type=bool, default=False, help='Use layer-wise exclusion loss')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for exclusion loss')
    parser.add_argument('--exclusion_weight', type=float, default=0.01, help='Weight for exclusion loss')
    parser.add_argument('--exc_layers_dist', type=int, default=2, help='Number of layers in between to calculate exclusion')
    args = parser.parse_args()

    if args.exclusion_loss and not args.interm_features_fc:
        args.exclusion_loss = False

    args.run_name = '{}_{}_image{}_batch{}_SGDlr{}_pt{}_pl{}_seed{}_epochs{}_{}_warmup{}_epochDecay{}_interFeatClassHead{}_excLoss{}_excWeight{}_excLayers{}'.format(
    args.dataset_name, args.model_name, args.image_size, args.batch_size, 
    args.learning_rate, args.pretrained, args.load_partial_mode, args.seed, 
    args.no_epochs, args.lr_scheduler, args.warmup_steps, args.epoch_decay, 
    args.interm_features_fc, args.exclusion_loss, args.exclusion_weight, args.exc_layers_dist)

    logger.info(args)

    os.makedirs(args.results_dir, exist_ok=True)

    train_main(logger, args)            

if __name__ == '__main__':
    main()

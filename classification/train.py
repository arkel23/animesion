import os
import logging
import argparse
import pandas as pd
from statistics import mean

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsummary import summary

import models.models as models
import data.datasets as datasets

logger = logging.getLogger(__name__)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    corr_list = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        corr_list.append(correct_k.item())
        #res.append(correct_k.mul_(100.0 / batch_size))
    return corr_list

def update_lr(optimizer, lr): 
    # For updating learning rate   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def data_loading(args, split):
    if args.model_name == 'resnet18' or args.model_name == 'resnet152':
        if split=='train':
            transform = transforms.Compose([
                transforms.Resize((args.image_size+32, args.image_size+32)),
                transforms.RandomCrop((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else:
        transform = None

    if args.dataset_name == 'moeImouto':    
        dataset = datasets.moeImouto(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFaces':
        dataset = datasets.danbooruFaces(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'cartoonFace':
        dataset = datasets.cartoonFace(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFull':
        dataset = datasets.danbooruFull(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)   
    
    dataset_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=8)


    return dataset, dataset_loader

def model_selection(args):
    # initiates model and loss     
    if args.model_name=='shallow':
        model = models.ShallowNet(args)
    elif args.model_name=='resnet18' or args.model_name=='resnet152':
        model = models.ResNet(args)
    else:
        model = models.VisionTransformer(args)
    return model

def train_main(logger, args):

    # makes results_dir if doesn't exist
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # dataloader and train/test datasets
    train_set, train_loader = data_loading(args, split='train')
    _, val_loader = data_loading(args, split='val')
    _, test_loader = data_loading(args, split='test')
    args.num_classes = train_set.num_classes
    classid_classname_dic = train_set.classes

    # model
    model = model_selection(args)
    print(str(model.configuration))
    #f.write(str(model.configuration))
    model.to(device)
    summary(model, input_size=iter(train_loader).next()[0].shape[1:])
    
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    #optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate)
    optimizer = torch.optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)
    
    # Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    train_loss_avg = []
    val_loss_avg = []
    top1_accuracies = []
    top5_accuracies = []
    best_epoch = 0
    curr_acc = 0
    top_acc = 0

    for epoch in range(args.train_epochs):
        model.train()
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

            # prints current set of results after each 10 iterations
            if (i % 10) == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}"
                    .format(epoch+1, args.train_epochs, i+1, total_step, loss.item()))
            current_losses.append(loss.item()) 
                
        # Decay learning rate
        if (epoch+1) % args.epoch_decay == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
            
        # calculates mean of losses for current epoch and appends to list of avgs
        train_loss_avg.append(mean(current_losses)) 

        # validates on test set once per epoch, calculates top1/5 acc and val loss avg
        curr_acc = validate(device=device, model=model, criterion=criterion, loader=val_loader,
        top1_accuracies=top1_accuracies, top5_accuracies=top5_accuracies, val_loss_avg=val_loss_avg)

        # Save the model checkpoint if the top1-acc is higher than current highest
        if curr_acc > top_acc:
            torch.save(model.state_dict(), os.path.join(results_dir, 
            '{}.ckpt'.format(args.name)))
            top_acc = curr_acc
            best_epoch = epoch + 1
        
    # validate on test set and plot results
    validate(device=device, model=model, criterion=criterion, loader=test_loader, 
    top1_accuracies=top1_accuracies, top5_accuracies=top5_accuracies)

    logger.info('Finished training successfully. Best val accuracy: {}, at epoch no: {}/{}'.format(
        top_acc, best_epoch, args.train_epochs))

    # contains the top1/5 accuracies for the validation after each epoch, and the last one for the test
    df_accuracies = pd.DataFrame(list(zip(top1_accuracies, top5_accuracies)))
    # contains the training and validation loss averages for each epoch
    df_losses = pd.DataFrame(list(zip(train_loss_avg, val_loss_avg)))
    df_accuracies.to_csv(os.path.join(results_dir, 
    '{}_accuracies.csv'.format(args.name)), sep=',', header=False, index=False)
    df_losses.to_csv(os.path.join(results_dir, 
    '{}_losses.csv'.format(args.name)), sep=',', header=False, index=False)

def validate(device, model, criterion, loader,
    top1_accuracies, top5_accuracies, val_loss_avg=[]):
    # Test the model (validation set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout probability goes to 0
    model.eval()
    with torch.no_grad():
        correct_1 = 0
        correct_5 = 0
        total = 0
        current_losses = []

        for images, labels in loader:
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
            
        # append avg val loss
        val_loss_avg.append(mean(current_losses))

        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1/total
        top1_accuracies.append(curr_top1_acc)
        print('Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %'.format(curr_top1_acc))
        curr_top5_acc = 100 * correct_5/total
        top5_accuracies.append(curr_top5_acc)
        print('Val/Test Top-5 Accuracy of the model on the test images: {:.4f} %'.format(curr_top5_acc))

        return curr_top1_acc

def main():

    logging.basicConfig(filename='logs.txt', level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFaces", "cartoonFace", "danbooruFull"], 
                        default="moeImouto", help="Which dataset to use.")
    parser.add_argument("--dataset_path", required=True,
                        help="Path for the dataset.")
    parser.add_argument("--model_name", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32', 'H_14',
                        'B_16_imagenet1k', 'B_32_imagenet1k', 
                        'L_16_imagenet1k', 'L_32_imagenet1k'],
                        default="shallow",
                        help="Which model architecture to use")
    parser.add_argument("--results_dir", default="results_training", type=str,
                        help="The directory where results will be stored")
    parser.add_argument("--image_size", default=224, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batch size for train/val/test.")
    parser.add_argument("--train_epochs", default=200, type=int,
                        help="Total number of epochs for training.")                         
    parser.add_argument("--epoch_decay", default=50, type=int,
                        help="After how many epochs to decay the learning rate once.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Initial learning rate.")  
    parser.add_argument("--pretrained", type=bool, default=False,
                        help="For models with pretrained weights available"
                        "Default=False")
    parser.add_argument("--checkpoint_path", type=str, 
                        default=None)     
    parser.add_argument("--transfer_learning", type=bool, default=False,
                        help="Load partial state dict for transfer learning"
                        "Resets the [embeddings, logits and] fc layer for ViT"
                        "Resets the fc layer for Resnets"
                        "Default=False")    
    parser.add_argument("--load_partial_mode", choices=['full_tokenizer', 'patchprojection', 'posembeddings', 'clstoken', 
        'patchandposembeddings', 'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                        help="Load pre-processing components to speed up training")        
    args = parser.parse_args()

    logger.info(args)

    train_main(logger, args)            

if __name__ == '__main__':
    main()

import os
import logging
import argparse
from statistics import mean

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsummary import summary

import models.models as models
import data.datasets as datasets

logger = logging.getLogger(__name__)

def update_lr(optimizer, lr): 
    # For updating learning rate   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def data_loading(args, train):
    if args.model_type == 'resnet18' or args.model_type == 'resnet152':
        if train:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
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
        input_size=args.image_size, train=train, transform=transform)
    elif args.dataset_name == 'danbooruFacesCrops':
        pass
        '''
        # NEEDS UPDATING
        dataset = datasets.danbooruFacesCrops(root=args.dataset_path,
        input_size=args.image_size, train=train, transform=transform)
        '''
    dataset_loader = data.DataLoader(dataset,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=4)

    return dataset, dataset_loader

def model_selection(args, no_classes):
    # initiates model and loss     
    if args.model_type=='shallow':
        model = models.ShallowNet(no_classes)
    elif args.model_type=='resnet18' or args.model_type=='resnet152':
        model = models.ResNet(no_classes, args)
    else:
        model = models.VisionTransformer(no_classes, args)
    return model

def train_main(logger, args):

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Controlling source of randomness: pytorch RNG
    torch.manual_seed(0)

    # dataloader and train/test datasets
    train_set, train_loader = data_loading(args, train=True)
    test_set, test_loader = data_loading(args, train=False)
    no_classes = train_set.no_classes

    # model
    model = model_selection(args, no_classes)
    model.to(device)
    # prints model summary (layers, parameters by giving it a sample input)
    summary(model, input_size=iter(train_loader).next()[0].shape[1:])
    
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate)
    #optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    training_proc_avg = []
    test_proc_avg = []

    for epoch in range(args.train_epochs):
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

            if (i % args.checkpoint_each_epochs) == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}"
                    .format(epoch+1, args.train_epochs, i+1, total_step, loss.item()))
                # appends the current value of the loss into a list
                current_losses.append(loss.item()) 
                
        # Decay learning rate
        if (epoch+1) % args.epoch_decay == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
            
        # calculates mean of losses for current epoch and appends to list of avgs
        training_proc_avg.append(mean(current_losses)) 

        # validates on test set once per epoch (or a few no of epochs)
        # TO DO: ADD EARLY STOPPING IF TEST_PROC_AVG[current]>test_proc[prev] break;
        validate(device=device, batch_size=args.batch_size, 
        model=model, criterion=criterion, no_classes=no_classes, 
        training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
        loader=test_loader, dataset_name=args.dataset_name, last=False)

    # validate on test set and plot results
    validate(device=device, batch_size=args.batch_size, 
    model=model, criterion=criterion, no_classes=no_classes, 
    training_proc_avg=training_proc_avg, test_proc_avg=test_proc_avg, 
    loader=test_loader, dataset_name=args.dataset_name, last=True)

    # Save the model checkpoint
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    torch.save(model.state_dict(), os.path.join(results_dir, 
        'model={}_pretrained={}.ckpt'.format(
        args.model_type, args.pretrained)))

    logger.info('Finished training successfully.')
    
def validate(device, batch_size, model, criterion, no_classes, 
    training_proc_avg, test_proc_avg, loader, dataset_name=False, last=False):
    # Test the model (validation set)
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # dropout doesn't
    model.eval()  
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

def main():

    logging.basicConfig(filename='logs.txt', level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFacesCrops"], 
                        default="moeImouto", help="Which dataset to use.")
    parser.add_argument("--dataset_path", required=True,
                        help="Path for the dataset.")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32', 'H_14',
                        'B_16_imagenet1k', 'B_32_imagenet1k', 
                        'L_16_imagenet1k', 'L_32_imagenet1k'],
                        default="shallow",
                        help="Which model architecture to use")
    parser.add_argument("--results_dir", default="results", type=str,
                        help="The directory where results will be stored")
    parser.add_argument("--image_size", default=224, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for train/val/test.")
    parser.add_argument("--train_epochs", default=200, type=int,
                        help="Total number of epochs for training.")                         
    parser.add_argument("--checkpoint_each_epochs", default=5, type=int,
                        help="Run prediction on validation set every so many epochs."
                        "Also saves checkpoints at this value."
                        "Will always run one test at the end of training.")
    parser.add_argument("--epoch_decay", default=50, type=int,
                        help="After how many epochs to decay the learning rate once.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Initial learning rate.")  
    parser.add_argument("--pretrained", type=bool, default=False,
                        help="For models with pretrained weights available"
                        "Default=False"
                        "If inputs anything (even the flag!) will take as True")                      
    args = parser.parse_args()

    logger.info(args)

    train_main(logger, args)            

if __name__ == '__main__':
    main()

from statistics import mean

import torch
import wandb

import utilities.utilities as utilities

def train_one_epoch(args, f, epoch, model, device, 
    criterion, optimizer, lr_scheduler, train_loader, train_loss_avg):
    
    model.train()
    current_losses = []
    steps_per_epoch = len(train_loader)

    for i, batch in enumerate(train_loader):

        if args.multimodal:
            images, labels, captions = batch
            captions = captions.squeeze(dim=1).to(device)
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
    
        # Forward pass
        if args.multimodal and args.exclusion_loss:
            outputs, exclusion_loss = model(images, captions)
        elif args.multimodal:
            outputs = model(images, captions)
        elif args.exclusion_loss:
            outputs, exclusion_loss = model(images)
        else:
            outputs = model(images)
        
        loss_class = criterion(outputs, labels)
        if args.exclusion_loss:
            loss = loss_class - (args.exclusion_weight * exclusion_loss)
        else:
            loss = loss_class
            
        # Backward and optimize
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        optimizer.zero_grad()
        
        current_losses.append(loss.item()) 

        # prints current set of results after each args.log_freq iterations
        if (i % args.log_freq) == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            curr_line = "Epoch [{}/{}], Step [{}/{}] Loss: {:.8f}, LR: {:.8f}\n".format(
                epoch+1, args.no_epochs, i+1, steps_per_epoch, loss.item(), curr_lr)
            utilities.print_write(f, curr_line)
            wandb.log({'Training loss (step)': loss.item(),
                'Learning rate (current)': curr_lr})

        if args.debugging and ((i + 1) % (args.log_freq * 3) == 0):
            break    

    # Decay learning rate
    if not lr_scheduler:
        if (epoch+1) % args.epoch_decay == 0:
            utilities.update_lr(optimizer)

    # calculates mean of losses for current epoch and appends to list of avgs
    train_loss_avg.append(mean(current_losses))
    wandb.log({'Training loss (epoch)': mean(current_losses)}) 

    return 0


def validate(args, f, device, model, criterion, loader,
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
        steps_per_epoch = len(loader)

        for i, batch in enumerate(loader):
            if args.multimodal:
                images, labels, captions = batch
                captions = captions.squeeze(dim=1).to(device)
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
                
            # Forward pass
            if args.multimodal and args.exclusion_loss:
                outputs, exclusion_loss = model(images, captions)
            elif args.multimodal:
                outputs = model(images, captions)
            elif args.exclusion_loss:
                outputs, exclusion_loss = model(images)
            else:
                outputs = model(images)
            
            loss_class = criterion(outputs, labels)
            if args.exclusion_loss:
                loss = loss_class - (args.exclusion_weight * (args.temperature ** 2) * exclusion_loss)
            else:
                loss = loss_class    
            current_losses.append(loss.item())
            
            # calculate top-k (1 and 5) accuracy
            total += labels.size(0)
            curr_corr_list = utilities.accuracy(outputs.data, labels, (1, 5, ))
            correct_1 += curr_corr_list[0]
            correct_5 += curr_corr_list[1]

            if i % args.log_freq == 0:
                curr_line = "Validation/Test Step [{}/{}] Loss: {:.8f}\n".format(
                i+1, steps_per_epoch, loss.item())
                utilities.print_write(f, curr_line)

            if args.debugging and ((i + 1) % (args.log_freq * 3) == 0):
                break    
            
        # append avg val loss
        val_loss_avg.append(mean(current_losses))

        # compute epoch accuracy in percentages
        curr_top1_acc = 100 * correct_1/total
        top1_accuracies.append(curr_top1_acc)
        curr_line = 'Val/Test Top-1 Accuracy of the model on the test images: {:.4f} %'.format(curr_top1_acc)
        utilities.print_write(f, curr_line)
        
        curr_top5_acc = 100 * correct_5/total
        top5_accuracies.append(curr_top5_acc)
        curr_line = 'Val/Test Top-5 Accuracy of the model on the test images: {:.4f} %'.format(curr_top5_acc)
        utilities.print_write(f, curr_line)

        wandb.log({"Epoch": len(top1_accuracies),
        "Val Accuracy Top-1": curr_top1_acc, 
        "Val Accuracy Top-5": curr_top5_acc,
        "Val Loss": mean(current_losses)})

        return curr_top1_acc

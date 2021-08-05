import math
import random
import numpy as np
import matplotlib.pyplot as plt 
import cv2

import torch
from torch.optim.lr_scheduler import LambdaLR

def print_write(f, line):
    f.write('{}\n'.format(line))
    print(line)

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

def update_lr(optimizer): 
    # For updating learning rate by 1/3 of current value
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 3

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
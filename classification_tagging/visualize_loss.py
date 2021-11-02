import os
import copy
import argparse
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import loss_landscapes
import loss_landscapes.metrics

from train import environment_loader
import utilities as utilities

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#matplotlib.rcParams['figure.figsize'] = [18, 12]


def plot_1d(loss_data, steps, out_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1/steps * i for i in range(steps)], loss_data)
    ax.set_title('Linear Interpolation of Loss')
    ax.set_xlabel('Interpolation Coefficient')
    ax.set_ylabel('Loss')
    fig.tight_layout()
    fig.savefig('{}'.format(out_name), dpi=300)
    print('{} plot saved.'.format(out_name))


def plot_2d(loss_data_fin, levels, out_name):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    CS = ax.contour(loss_data_fin, levels=levels)
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_title('Loss Contours around Trained Model')
    fig.tight_layout()
    fig.savefig('{}'.format(out_name), dpi=300)
    print('{} plot saved.'.format(out_name))


def plot_3d(loss_data_fin, steps, out_name):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(20, 30)
    ax.set_title('Surface Plot of Loss Landscape')
    fig.tight_layout()
    fig.savefig('{}'.format(out_name), dpi=300)
    print('{} plot saved.'.format(out_name))


def visualize_loss_landscape(args):

    (device, train_set, train_loader, val_loader, test_loader,
    classid_classname_dic, model, optimizer, lr_scheduler,
    mask_scheduler, tokenizer) = environment_loader(args, init=False)
    model_final = copy.deepcopy(model)

    args.checkpoint_path = None
    (device, train_set, train_loader, val_loader, test_loader,
    classid_classname_dic, model, optimizer, lr_scheduler,
    mask_scheduler, tokenizer) = environment_loader(args, init=False)
    model_initial = copy.deepcopy(model)

    criterion = torch.nn.CrossEntropyLoss()

    # data that the evaluator will use when evaluating loss
    loader = train_loader if args.split == 'train' else test_loader
    x, y = iter(loader).__next__()
    x = x.to(device)
    y = y.to(device)
    
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    # compute loss data
    loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, metric, args.steps, deepcopy_model=True)
    # 1d linear interpolation between two points
    plot_1d(loss_data, args.steps, out_name=os.path.join(args.save_dir, '1d.png'))

    # planar approximation of loss around a point
    loss_data_fin = loss_landscapes.random_plane(model_final, metric, args.distance, args.steps, normalization='filter', deepcopy_model=True)
    # contour plot
    plot_2d(loss_data_fin, args.levels, out_name=os.path.join(args.save_dir, '2d.png'))
    # surface plot
    plot_3d(loss_data_fin, args.steps, out_name=os.path.join(args.save_dir, '3d.png'))
    

def main():
  
    parent_parser = utilities.misc.ret_args(ret_parser=True)

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--split", type=str, default='test', choices=['train', 'test'],
        help='train/val/test loader for loss')
    parser.add_argument("--steps", type=int, default=100, help='interpolation resolution')
    parser.add_argument("--distance", type=int, default=3, help='distance for 2d loss')
    parser.add_argument("--levels", type=int, default=20, help='levels for contour plot')
    parser.set_defaults(results_dir='results_vis', no_epochs=1)
    args = parser.parse_args()

    save_dir = 'steps{}dist{}bs{}_{}'.format(args.steps, args.distance, args.batch_size, 
        os.path.basename(os.path.normpath(os.path.splitext(args.checkpoint_path)[0])))
    args.save_dir = os.path.join(args.results_dir, save_dir)
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)

    visualize_loss_landscape(args)
    

if __name__ == '__main__':
    main()
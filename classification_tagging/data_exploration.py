import re
import os
import glob
import argparse
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import utilities as utilities
 
def imshow(inp, out_name, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.uint8(np.clip(inp, 0, 1) * 255)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=4, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}'.format(out_name), dpi=300)


def load_dataset(args, split=None):

    if split==None:
        split = args.split

    # set all images to a certain size for visualization purposes
    img_size = args.image_size
    transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()])

    if args.dataset_name == 'moeImouto':    
        dataset = utilities.data_selection.moeImouto(args, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFaces' or args.dataset_name == 'danbooruFull':
        dataset = utilities.data_selection.danbooruFacesFull(args, split=split, transform=transform)
	
    return dataset


def data_visualization(args):
    
    dataset = load_dataset(args)

    if args.data_vis_full:
        dataset_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=4)
    else:
        dataset_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4)

    num_classes = dataset.num_classes
    classid_classname_dic = dataset.classes
    print('Total number of images: {}, Total number of classes: {}'.format(
    len(dataset), num_classes))

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    nrows = math.floor(math.sqrt(args.batch_size))

    if args.data_vis_full:
        for i, (images, labels) in enumerate(dataset_loader):
            out_name = os.path.join(args.results_dir, '{}.png'.format(i))
            if args.labels:
                # Make a grid from batch
                class_unique = []
                for x in labels:
                    current_class = classid_classname_dic.loc[classid_classname_dic['class_id']==x.item()]['class_name'].item()
                    if current_class not in class_unique:
                        class_unique.append(current_class)

                grid = torchvision.utils.make_grid(images, nrow=nrows)
                imshow(grid, out_name, title=class_unique)
            else:
                torchvision.utils.save_image(images, out_name, nrow=nrows)
            
        print('Finished saving images.')
        video_from_frames(args)
        print('Finished writing video.')

    else:
        images, labels = iter(dataset_loader).next()
        out_name = os.path.join(args.results_dir, 
        '{}_{}_labels{}.png'.format(args.dataset_name, args.split, args.labels))
        # Make a grid from batch
        grid = torchvision.utils.make_grid(images, nrow=nrows)
        if args.labels:
            class_list = []
            i = 0
            for x in labels:
                current_class = classid_classname_dic.loc[classid_classname_dic['class_id']==x.item()]['class_name'].item()
                class_list.append(current_class)
                i += 1
                if i%nrows == 0:
                    class_list.append('\n')
            class_list = '  '.join(class_list)
            # displays classes names
            imshow(grid, out_name, title=class_list)
        else:
            # just displays the images
            torchvision.utils.save_image(grid, out_name, nrow=nrows)


def video_from_frames(args):
    images_paths = glob.glob(os.path.join(args.results_dir, '*.png'))
    images_paths = [p for p in images_paths if re.search("[0-9]+(.png)", p)]
    no_images = len(images_paths)
    print(no_images)
    img_sample = cv2.imread(images_paths[0])
    print(img_sample.shape)
    
    height, width = img_sample.shape[0:2]
    video_out_name = os.path.join(args.results_dir, 
    '{}_{}_labels{}.mp4'.format(args.dataset_name, args.split, args.labels))
    video_out = cv2.VideoWriter(video_out_name, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (width, height))
    for i in range(no_images):
        curr_path = os.path.join(args.results_dir, '{}.png'.format(i))
        img = cv2.imread(curr_path)
        video_out.write(img)
        os.remove(curr_path)
    video_out.release()


def data_stats(args):
    
    if args.stats_partial:
        dataset = load_dataset(args)
        no_samples = len(dataset)
        df = dataset.df
    else:
        if args.dataset_name=='moeImouto':
            dataset_train = load_dataset(args, split='train')
            dataset_test = load_dataset(args, split='test')
            no_samples = len(dataset_train) + len(dataset_test)
            df_train = dataset_train.df
            df_test = dataset_test.df
            df = pd.concat([df_train, df_test])
        elif args.dataset_name == 'danbooruFaces' or args.dataset_name == 'danbooruFull':
            dataset_train = load_dataset(args, split='train')
            dataset_val = load_dataset(args, split='val')
            dataset_test = load_dataset(args, split='test')
            dataset = dataset_train
            no_samples = len(dataset_train) + len(dataset_val) + len(dataset_test)
            df_train = dataset_train.df
            df_val = dataset_val.df
            df_test = dataset_test.df
            df = pd.concat([df_train, df_val, df_test])

    classid_classname_dic = dataset.classes
    num_classes = dataset.num_classes

    samples_per_class = df.groupby('class_id', as_index=True).count()['dir'].squeeze()
    
    set_mean = samples_per_class.mean()
    set_median = samples_per_class.median()
    set_std = samples_per_class.std()
    print('Dataset: {}\n'
    'Total number of samples: {}\n'
	'Number of classes: {}\n'
    'Mean number of samples per class: {}\n'
	'Median number of samples per class: {}\n'
	'Standard deviation of samples per class: {}'.format(
	args.dataset_name, no_samples, num_classes, set_mean, set_median, set_std))

    samples_per_class_ordered = samples_per_class.sort_values(0, ascending=False)
    print(samples_per_class_ordered.head())
	
    print('Characters with most number of samples: ')
    for i in range(10):
            current_id = samples_per_class_ordered.index[i]
            current_class = classid_classname_dic.loc[classid_classname_dic['class_id']==current_id]['class_name'].item()
            print('No. {}: {} (Class ID: {}) with {} samples'.format(
            i+1, current_class, current_id, samples_per_class_ordered.iloc[i]
            ))
	
    print('Characters with least number of samples: ')
    for i in range(10):
            current_id = samples_per_class_ordered.index[-1-i]
            current_class = classid_classname_dic.loc[classid_classname_dic['class_id']==current_id]['class_name'].item()
            print('No. {}: {} (Class ID: {}) with {} samples'.format(
            i+1, current_class, current_id, samples_per_class_ordered.iloc[-1-i]
            ))

    fig, axs = plt.subplots(1)
    fig.suptitle('Histogram of Classes for {} Dataset'.format(args.dataset_name))

	# only plot first 100
    bins = 100
    axs.bar(np.arange(bins), samples_per_class_ordered.iloc[0:bins].to_numpy())
		
    axs.set_ylabel('No. of samples per class')
    axs.set_title('Ordered based on no. of samples')

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    file_name = os.path.join(args.results_dir, 'histogram_{}.png'.format(args.dataset_name))	
    fig.savefig(file_name, dpi=300)


def main():
    torch.manual_seed(0)
    parent_parser = utilities.misc.ret_args(ret_parser=True)

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--split", default='test', type=str,
                        help="Split to visualize")
    parser.add_argument("--data_vis_full", action='store_true',
                        help="Save all images into a video.")  
    parser.add_argument("--labels", action='store_true',
                        help="Include labels as title during the visualization video (requires a LOT more time).")
    parser.add_argument("--display_images", action='store_true',
                        help="If False skips to data_stats function else display images as single plot or video.")
    parser.add_argument("--stats_partial", action='store_true',
                        help="If true will display stats for a certain subset instead of the whole.")
    parser.set_defaults(results_dir='data_exploration')
    args = parser.parse_args()
    assert args.dataset_path, "Requires to input --dataset_path"

    if args.display_images:
        data_visualization(args)
    data_stats(args)


if __name__ == '__main__':
    main()
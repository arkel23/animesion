import os
import pickle
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torchvision
import torchvision.transforms as transforms

from train import environment_loader
import utilities as utilities
from utilities.build_vocab import Vocabulary
 
DEFAULT_MAX_TEXT_SEQ_LEN = 16

def return_prepared_inputs(file_path, args, device, data_set, mask_scheduler=None):
    transform = data_set.transform
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image = transform(Image.open(file_path)).to(device).unsqueeze(0)

    if args.mode == 'recognition_tagging' or args.mode == 'generate_tags':
        if args.masking_behavior == 'constant':
            text_prompt = torch.ones((1, args.max_text_seq_len), dtype=torch.int64).to(device)
        else:
            text_prompt = torch.randint(0, mask_scheduler.vocab_size-1, (1, args.max_text_seq_len)).to(device)
        text_prompt[:, 0] = mask_scheduler.special_tokens[1]
        text_prompt[:, -1] = mask_scheduler.special_tokens[2]
        return image, text_prompt
    
    return image


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
    plt.pause(5)
    if save_results:
        plt.savefig('{}'.format(out_name), dpi=300)
    plt.close()


def forward_vision(args, classid_classname_dic, model, image, file_path):

    # forward pass plus printing and plotting classification results
    print(file_path)
    file_name_no_ext = os.path.splitext(os.path.split(file_path)[1])[0]
    out_name = os.path.join(args.results_dir, '{}.jpg'.format(file_name_no_ext))
        
    with torch.no_grad():
        if args.ret_attn_scores:
            outputs, att_mat = model(image)
            outputs = outputs.squeeze(0)
            utilities.vis_attention(args, image, outputs, att_mat, file_name_no_ext)
            
            classes_predicted = []
            classes_predicted.append(file_name_no_ext)
            classes_predicted.append('\n')
            for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                prob = torch.softmax(outputs, -1)[idx].item() * 100
                class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                classes_predicted.append(predict_text)
                print(predict_text, end='')
                    
            classes_predicted = '  '.join(classes_predicted)
            grid = torchvision.utils.make_grid(image)
            imshow(grid, out_name, title=classes_predicted, save_results=args.save_results)
                    
        else:
            outputs = model(image).squeeze(0)
            classes_predicted = []
            classes_predicted.append(file_name_no_ext)
            classes_predicted.append('\n')
            for i, idx in enumerate(torch.topk(outputs, k=5).indices.tolist()):
                prob = torch.softmax(outputs, -1)[idx].item() * 100
                class_name = classid_classname_dic.loc[classid_classname_dic['class_id']==idx, 'class_name'].item()
                predict_text = 'Prediction No. {}: {} [ID: {}], Confidence: {}\n'.format(i+1, class_name, idx, prob)
                classes_predicted.append(predict_text)
                print(predict_text, end='')

            classes_predicted = '  '.join(classes_predicted)
            grid = torchvision.utils.make_grid(image)
            imshow(grid, out_name, title=classes_predicted, save_results=args.save_results)


def forward_multimodal(args, model, tokenizer, voc, image, text_prompt, file_path):
    
    print(file_path)

    with torch.no_grad():
        out_cls, out_tokens_text = model(image, text=text_prompt)
        
    text_prob, text_pred = torch.topk(out_tokens_text, k=1, dim=2, largest=True, sorted=True)
    text_pred = text_pred.squeeze()
    #print(text_prompt)
    #print(text_prob)
    decoded_text = tokenizer.decode(text_pred)
    if args.tokenizer == 'tag':
        gen_tags = sorted({tag for tag in decoded_text if tag in voc.word2idx.keys()})
        print('Predicted {} tags: {}'.format(len(gen_tags), gen_tags))
    else:
        gen_tags = sorted({tag for tag in voc.word2idx.keys() if tag in decoded_text})
        print('Predicted {} tags: {}'.format(len(gen_tags), gen_tags))


def recognition_vision(args, device, data_set, model):
    classid_classname_dic = data_set.classes
    
    # Images to be tested
    file_list = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path) if os.path.isfile(
        os.path.join(args.test_path, f))]
    
    model.eval()
    for file_path in file_list:
        image = return_prepared_inputs(file_path, args, device, data_set)
        forward_vision(args, classid_classname_dic, model, image, file_path)


def recognition_tagging(args, device, data_set, model, mask_scheduler, tokenizer):

    if args.tokenizer == 'tag':
        voc = tokenizer.vocab
    else:
        with open(os.path.join(args.dataset_path, 'labels', 'vocab.pkl'), 'rb') as f:
            voc = pickle.load(f)

    model.eval()

    file_list = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path) if os.path.isfile(
        os.path.join(args.test_path, f))]

    for file_path in file_list:
        image, text_prompt = return_prepared_inputs(file_path, args, device, data_set, mask_scheduler)
        forward_multimodal(args, model, tokenizer, voc, image, text_prompt, file_path)


def generate_tags_df(args, device, data_set, model, mask_scheduler, tokenizer):

    path_root, filename = os.path.split(args.test_path)
    filename_no_ext = os.path.splitext(filename)[0]
    new_filename = os.path.join(path_root, '{}_tags.csv'.format(filename_no_ext))
    
    df = pd.read_csv(args.test_path,  sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})
    df['tags_cat0'] = ''

    if args.tokenizer == 'tag':
        voc = tokenizer.vocab
    else:
        with open(os.path.join(args.dataset_path, 'labels', 'vocab.pkl'), 'rb') as f:
            voc = pickle.load(f)

    model.eval()

    for i, file_path in enumerate(df.dir):
        image, text_prompt = return_prepared_inputs(
            os.path.join(path_root, 'data', file_path), args, device, data_set, mask_scheduler)

        with torch.no_grad():
            out_cls, out_tokens_text = model(image, text=text_prompt)
        
        text_prob, text_pred = torch.topk(out_tokens_text, k=1, dim=2, largest=True, sorted=True)
        text_pred = text_pred.squeeze()
        
        decoded_text = tokenizer.decode(text_pred)
        if args.tokenizer == 'tag':
            gen_tags = sorted({tag for tag in decoded_text if tag in voc.word2idx.keys()})
        else:
            gen_tags = sorted({tag for tag in voc.word2idx.keys() if tag in decoded_text})

        df.at[i, 'tags_cat0'] = gen_tags

        if i % 500 == 0:
            print('{}/{}: {}: {}'.format(i, len(df), file_path, gen_tags))

    df.to_csv(new_filename, header=True, index=False)
    print('Saved new dataset with tags')


def main():
    
    '''
    #device, model, data_set, data_loader = environment_loader(args)

    #os.makedirs(args.results_infer, exist_ok=True)
    #inference(args, device, model, data_set)           
    '''
    
    parent_parser = utilities.misc.ret_args(ret_parser=True)

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--mode", choices=['recognition_vision', 'recognition_tagging', 'generate_tags'], type=str, 
                        default='recognition_vision', help="Mode for inference (multimodal or vision).")
    parser.add_argument("--test_path", type=str, default='test_images/',
                        help="The directory where test image is stored.")
    parser.add_argument("--results_infer", default="results_inference", type=str,
                        help="The directory where inference results will be stored.")
    parser.add_argument("--save_results", type=bool, default=False,
                        help="Save the images after transform and with label results.")   
    args = parser.parse_args()

    if args.mode == 'recognition_tagging' or args.mode == 'generate_tags':
        args.multimodal = True
        args.mask_schedule = 'full'
        if not args.max_text_seq_len:
            args.max_text_seq_len = DEFAULT_MAX_TEXT_SEQ_LEN
    
    assert args.dataset_path, "Requires to input --dataset_path"
    
    (device, train_set, train_loader, val_loader, test_loader,
    classid_classname_dic, model, optimizer, lr_scheduler,
    mask_scheduler, tokenizer) = environment_loader(args, init=False)

    if args.mode == 'recognition_vision':
        recognition_vision(args, device, train_set, model)
    elif args.mode == 'recognition_tagging':
        recognition_tagging(args, device, train_set, model, mask_scheduler, tokenizer)
    elif args.mode == 'generate_tags':
        generate_tags_df(args, device, train_set, model, mask_scheduler, tokenizer)


if __name__ == '__main__':
    main()
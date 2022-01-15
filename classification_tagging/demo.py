import os
import pickle
import argparse
import gradio as gr

import torch
import torchvision
import torchvision.transforms as transforms

from train import environment_loader
from inference import return_prepared_inputs, forward_multimodal
import utilities as utilities
from utilities.build_vocab import Vocabulary
 
DEFAULT_MAX_TEXT_SEQ_LEN = 16

def demo(file_path):
    
    file_path = str(file_path.name)
    classid_classname_dic = train_set.classes

    if args.tokenizer == 'tag':
        voc = tokenizer.vocab
    else:
        with open(os.path.join(args.dataset_path, 'labels', 'vocab.pkl'), 'rb') as f:
            voc = pickle.load(f)

    model.eval()

    image, text_prompt = return_prepared_inputs(file_path, args, device, train_set, mask_scheduler)
    output = forward_multimodal(args, classid_classname_dic, model, tokenizer, voc, 
        image, text_prompt, file_path, print_local=False)
    print(output)
    return output

parent_parser = utilities.misc.ret_args(ret_parser=True)

parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
parser.add_argument("--test_path", type=str, default='test_images/',
                        help="The directory where test image is stored.")
args = parser.parse_args()

args.mode = 'recognition_tagging'
args.results_infer = 'results_inference'
args.multimodal = True
args.mask_schedule = 'full'
if not args.max_text_seq_len:
    args.max_text_seq_len = DEFAULT_MAX_TEXT_SEQ_LEN
    
assert args.dataset_path, "Requires to input --dataset_path"
    
(device, train_set, train_loader, val_loader, test_loader,
classid_classname_dic, model, optimizer, lr_scheduler,
mask_scheduler, tokenizer) = environment_loader(args, init=False)

title = 'Animesion'
description = 'A Framework for Anime Character Recognition'
article = '''<p style='text-align: center'>
    <a href='https://arxiv.org/abs/2101.08674'>
    DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset For Anime Character Recognition</a> | 
    <a href='https://github.com/arkel23/animesion/'>GitHub Repo</a></p>'''

#demo()
inputs = gr.inputs.Image(type='file', label='Input image')
outputs = gr.outputs.Textbox(type='auto', label='Predicted class and tags')
examples = [['test_images/homura_top.jpeg'], ['test_images/eren_face.jfif']]

gr.Interface(demo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True,share=False)
    

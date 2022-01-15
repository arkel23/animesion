import os
import pickle
import argparse
import gradio as gr

import torch
import torchvision
import torchvision.transforms as transforms

from train import environment_loader
from inference import return_prepared_inputs, forward_vision
import utilities as utilities


def demo(file_path):
    file_path = str(file_path.name)
    classid_classname_dic = train_set.classes

    model.eval()

    image = return_prepared_inputs(file_path, args, device, train_set, mask_scheduler)
    output = forward_vision(args, classid_classname_dic, model, image, file_path, print_local=False)
    print(output)
    return output


parent_parser = utilities.misc.ret_args(ret_parser=True)

parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
parser.add_argument("--test_path", type=str, default='test_images/',
                    help="The directory where test image is stored.")
args = parser.parse_args()

args.mode = 'recognition_vision'
args.results_infer = 'results_inference'

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

inputs = gr.inputs.Image(type='file', label='Input image')
outputs = gr.outputs.Textbox(type='auto', label='Predicted class and tags')
examples = [['test_images/homura_top.jpeg'], ['test_images/eren_face.jfif']]

gr.Interface(demo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True, share=True) 

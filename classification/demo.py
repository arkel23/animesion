
import argparse
from PIL import Image
from torchvision import transforms
import gradio as gr
import torch

import models.models as models

def ret_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='B_16')
    parser.add_argument('--num_classes', type=int, default=3263)
    parser.add_argument('--checkpoint_path', type=str, 
    default='checkpoints/danbooruFaces_b16_ptTrue_batch16_imageSize128_50epochs_epochDecay20.ckpt')

    args = parser.parse_args()
    args.load_partial_mode = False
    args.transfer_learning = False
    args.interm_features_fc = False
    args.multimodal = False
    args.max_text_seq_len = None
    args.pretrained=True
    return args

args = ret_args()

#torch.hub.download_url_to_file('https://miro.medium.com/max/1400/1*aY5OzkUm_bfO9I8jltR9DA.jpeg', 'homura.jpg')
#torch.hub.download_url_to_file('https://static.wikia.nocookie.net/characters/images/5/5d/Eren-Jaeger.jpg/revision/latest/top-crop/width/360/height/450?cb=20160708134212', 'eren.jpg')

#dataset = datasets.data_loading(
#num_classes = 3263 #dataset.num_classes

model = models.model_selection(args, device='cpu')
model.eval()

tfms = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def recognize_img(img):
    inputs = tfms(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)#.squeeze(0)
        print(outputs.shape)
        outputs = outputs.squeeze(0)
        print(outputs.shape)

    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        predictions = '{}. Probability: {}'.format(idx, prob*100)

    return predictions

inputs = gr.inputs.Image(type='pil', label='Original image')
outputs = gr.outputs.Label(type='auto', label='Predicted class')

title = 'Animesion'
description = 'A Framework for Anime Character Recognition'
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2101.08674'>DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset For Anime Character Recognition</a> | <a href='https://github.com/arkel23/animesion/tree/main/classification'>GitHub Repo</a></p>"

examples = [['test_images/homura_top.jpeg'], ['test_images/eren_face.jfif']]

recognize_img(img=Image.open(examples[0][0]).convert('RGB'))
#gr.Interface(recognize_img, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(share=True)

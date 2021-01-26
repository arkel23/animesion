# Overview 
Anime character recognition/classification using PyTorch.

Our best model, ViT L-16 with image size 128x128 and batch size 64 achieves to get 85.95% and 94.23% top-1 and top-5 classification accuracies, among 3263 characters, compared to the best CNN model (ResNet-18) that only achieved 69.09% and 84.64%, respectively.

We hope that this work inspires other researchers to follow and build upon this path. ViT models have interesting properties for domain transfer that haven't been studied, and their big jump in terms of performance compared to CNNs suggest that they may be more suitable for drawn, sketched character recognition. This is due to the fact that CNNs are biased towards texture, and not shapes.

Checkpoints: [Google Drive](https://drive.google.com/drive/folders/15H1A5KoBQHQhtpFkCrKqwmvsfVYJY__1?usp=sharing). Note: I still need to upload some checkpoints.

# Features
* Variety of architectures to choose from: Shallow, Resnet18/152, ViT B-16/B-32/L-16/L-32
* Two datasets, moeImouto and DAF:re, with 173 and more than 3000 classes, respectively
* Pre-trained models for each of these architectures for image sizes of 128x128 or 224x224, trained for 50 or 200 epochs.
* Supporting scripts for making, visualization and stats for datasets.
* Scripts for training from scratch, evaluation (accuracy of a model with a certain set and pretrained weights), and inference (classifies all images in `test_images/` folder)

# To do
* Better documentation and organize the repo folder structure and files.
* Change train.py to log losses and accuracies into same file.
 
 # How to use
 The main scripts in this repo are the `train.py`, `evaluate.py` and `inference.py`. These are all supported by `models/models.py` and `data/datasets.py`.
 
 ## train.py
 
This script takes as input a certain of hyperparameters (dataset to use, model, batch and image size, among others) and trains the model, either from scratch, or from a checkpoint. If training from a checkpoint, it can also use it to do knowledge transfer between datasets, by for example using a checkpoint trained on *DAF:re* to classify images according to the characters in *moeImouto*.

Note: for ViT models it requires installing the Vision Transformer repository. It can be cloned and then installed through `pip install -e .` following instructions on either of these two repositories:
* https://github.com/arkel23/PyTorch-Pretrained-ViT
* https://github.com/lukemelas/PyTorch-Pretrained-ViT
The main difference is that for mine it allows to setup the Vision Transformer to keep the representation layer, a fully-connected layer used during pre-training, that's usually dropped when doing downstream tasks. If you're only doing inference the original [lukemelas](https://github.com/lukemelas/PyTorch-Pretrained-ViT) repository should be enough and more convenient, since it can be download and installed directly in one line `pip install pytorch_pretrained_vit`.

```
usage: train.py [-h] --name NAME [--dataset_name {moeImouto,danbooruFaces,cartoonFace}] --dataset_path DATASET_PATH
                [--model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32,H_14,B_16_imagenet1k,B_32_imagenet1k,L_16_imagenet1k,L_32_imagenet1k}] [--results_dir RESULTS_DIR]
                [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--train_epochs TRAIN_EPOCHS] [--epoch_decay EPOCH_DECAY] [--learning_rate LEARNING_RATE] [--pretrained PRETRAINED]
                [--checkpoint_path CHECKPOINT_PATH] [--transfer_learning TRANSFER_LEARNING]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of this run. Used for monitoring.
  --dataset_name {moeImouto,danbooruFaces,cartoonFace}
                        Which dataset to use.
  --dataset_path DATASET_PATH
                        Path for the dataset.
  --model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32,H_14,B_16_imagenet1k,B_32_imagenet1k,L_16_imagenet1k,L_32_imagenet1k}
                        Which model architecture to use
  --results_dir RESULTS_DIR
                        The directory where results will be stored
  --image_size IMAGE_SIZE
                        Image (square) resolution size
  --batch_size BATCH_SIZE
                        Batch size for train/val/test.
  --train_epochs TRAIN_EPOCHS
                        Total number of epochs for training.
  --epoch_decay EPOCH_DECAY
                        After how many epochs to decay the learning rate once.
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --pretrained PRETRAINED
                        For models with pretrained weights availableDefault=False
  --checkpoint_path CHECKPOINT_PATH
  --transfer_learning TRANSFER_LEARNING
                        Load partial state dict for transfer learningResets the [embeddings, logits and] fc layer for ViTResets the fc layer for ResnetsDefault=False
```

## evaluate.py

This script takes as input similar hyperparameters as the previous, but it puts the model into evaluation mode, where the weights are frozen and no gradients are calculated. Then, if given a checkpoint, it uses that checkpoint to evaluate the performance of the model on a certain dataset. It returns a series of stats, mostly the top-1 and top-5 accuracies,  for the whole batch, and optionally, the per-character accuracy.

```
usage: evaluate.py [-h] [--dataset_name {moeImouto,danbooruFaces}] [--dataset_path DATASET_PATH] [--image_size {128,224}] [--model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32}]
                   [--checkpoint_path CHECKPOINT_PATH] [--results_dir RESULTS_DIR] [--pretrained PRETRAINED] [--batch_size BATCH_SIZE] [--vis_arch VIS_ARCH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {moeImouto,danbooruFaces}
                        Which dataset to use (for no. of classes/loading model).
  --dataset_path DATASET_PATH
                        Path for the dataset.
  --image_size {128,224}
                        Image (square) resolution size
  --model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32}
                        Which model architecture to use
  --checkpoint_path CHECKPOINT_PATH
                        Path for model checkpoint to load.
  --results_dir RESULTS_DIR
                        The directory where results will be stored.
  --pretrained PRETRAINED
                        DON'T CHANGE! Always true since always loading when doing inference.
  --batch_size BATCH_SIZE
                        Batch size for train/val/test.
  --vis_arch VIS_ARCH   Visualize architecture through model summary.
```

## inference.py

This script also takes as input the model and image size to evaluate on, the dataset according to which classify, and the checkpoint. Then, it goes through all the RGB (it skips gray-scale, RGBa, etc.) images in the `test_images/` folder and outputs to the command line the top-5 classes it predicts, along with their confidence. 

```
usage: inference.py [-h] [--dataset_name {moeImouto,danbooruFaces}] [--dataset_path DATASET_PATH] [--images_path IMAGES_PATH] [--image_size {128,224}]
                    [--model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32}] [--checkpoint_path CHECKPOINT_PATH] [--results_dir RESULTS_DIR] [--pretrained PRETRAINED]
                    [--batch_size BATCH_SIZE] [--save_results SAVE_RESULTS]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {moeImouto,danbooruFaces}
                        Which dataset to use (for no. of classes/loading model).
  --dataset_path DATASET_PATH
                        Path for the dataset.
  --images_path IMAGES_PATH
                        Path for the images to be tested.
  --image_size {128,224}
                        Image (square) resolution size
  --model_type {shallow,resnet18,resnet152,B_16,B_32,L_16,L_32}
                        Which model architecture to use
  --checkpoint_path CHECKPOINT_PATH
                        Path for model checkpoint to load.
  --results_dir RESULTS_DIR
                        The directory where results will be stored.
  --pretrained PRETRAINED
                        DON'T CHANGE! Always true since always loading when doing inference.
  --batch_size BATCH_SIZE
                        Batch size for train/val/test. Just for loading the dataset.
  --save_results SAVE_RESULTS
                        Save the images after transform and with label results.
```

It can also, display these results along with the image in a matplotlib.pyplot window similar to what's seen below: 
![](https://github.com/arkel23/animesion/blob/main/classification/results_inference/homura_top.jpg)
![](https://github.com/arkel23/animesion/blob/main/classification/results_inference/kirito.jpg)
It not only works for anime images, but also works that take inspiration from anime, such as many videogames. 
![](https://github.com/arkel23/animesion/blob/main/classification/results_inference/dva.jpg)
And also for images that are not a face-crop:
![](https://github.com/arkel23/animesion/blob/main/classification/results_inference/rei_bodypillow.jpg)

To check if a particular character was in the dataset, take a look at this file with the classes names for [DAF:re](https://github.com/arkel23/animesion/blob/main/classification/data/danbooruFaces/classid_classname.csv) and [moeImouto](https://github.com/arkel23/animesion/blob/main/classification/data/moeImouto/classid_classname.csv)

We also tried it for people (and a cat), so hopefully I'll be able to post some results of that later. What came to me as interesting is that for the cat, it predicted `Naruto`, I guess based on the whiskers. Also, for a friend with curly hair, among the predictions included `Yamagishi Yukako`, a character with similar characteristics. 

![](https://github.com/arkel23/animesion/blob/main/classification/results_inference/muffin.jpg)

While the results were certainly far from perfect, this can serve as a basis for more studies on domain adaptation from natural images to sketches and drawn media.

# Extras

## data_exploration.py

Allows for visualization of the datasets, in terms of saving/displaying grids of the images, and outputting statistics such as mean, median and standard deviation for the different datasets. Also plots the histograms.

For a visualization of all the splits (along with the labels): [YouTube playlist](https://youtube.com/playlist?list=PLenBV8wMp2FyJHvBZM4FBxua7JggUUqvQ). In total, there's 6 videos, 3 for *DAF:re* and 3 for *moeImouto*.

A brief preview can be seen here for DAF:re and moeImouto, respectively:

![](https://j.gifs.com/ROpp10.gif)

![](https://j.gifs.com/XLyy5l.gif)

## models/models.py

Hosts the different models used for classification. Shallow is a shallow, 5 layer (2 convolutional + 2 fully-connected) network. ResNet-18/152 has been the basis for many CNN architectures and was SotA for image classification just a few years ago [(paper)](https://arxiv.org/abs/1512.03385). Vision Transformers [(ViT paper)](https://arxiv.org/abs/2010.11929) are the new SotA for image classification in many standard benchmarks such as ImageNet, among others. Their significance is that they forego convolutions completely, and rely only on self-attention. This allows ViT to attend to distant regions in the image, as it looks as the whole image as a sequence of patches, all at once. This is in comparison to CNNs which traditionally "look" at the image patch by patch, rendering them unable to grasp long-range dependencies.

## data/datasets.py

Holds the classes for the datasets. The structure is straightforward and allows for easily adding more datasets. I also included companion scripts to make the dataset dictionaries (file path with class) for any folder following similar structures in the following [repo](https://github.com/arkel23/custom_datadic_datasplit) 

## data/
### moeImouto
For the moeImouto dataset here's a sample of how the images look along with their classes. For the training and testing split files: [moeImouto repo](https://github.com/arkel23/moeimouto_animefacecharacterdataset)
![](https://github.com/arkel23/animesion/blob/main/classification/data_exploration/moeImouto_train.png)
Histogram of classes with most samples.
![](https://github.com/arkel23/animesion/blob/main/classification/data_exploration/histogram_moeImouto.png)
The dataset itself can be downloaded from [Kaggle](https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home) then stored in a folder containing the rest of the files, following the structure `moeImoutoDataset/data/`, where `data/` contains the folders containing images divided by class and the base folder `moeImoutoDataset/` contains the files included in the [repo](https://github.com/arkel23/moeimouto_animefacecharacterdataset) I described previously (`train.csv`, `test.csv`, and `classid_classname.csv`).

### DAF:re
Similarly, for DAF:re. Also, here's the repo for some more details on the dataset along with the files for training, testing and validation: [DAF:re repo](https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped)
![](https://github.com/arkel23/animesion/blob/main/classification/data_exploration/danbooruFaces_train.png)
Histogram of classes with most samples. It's clear that the distribution is very long-tailed.
![](https://github.com/arkel23/animesion/blob/main/classification/data_exploration/histogram_danbooruFaces.png)
The dataset itself can be downloaded using `rsync`:
```
rsync --verbose rsync://78.46.86.149:873/biggan/2019-07-27-grapeot-danbooru2018-animecharacterrecognition.tar ./
```
Then store in a folder containing the rest of the files, following the structure `dafreDataset/data/`, where `data/` contains the folders that have the images and the base folder `dafreDataset/` contains the files included in the [repo](https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped) I described previously (`train.csv`, `val.csv`, `test.csv`, and `classid_classname.csv`). Additionally, the folder names have to be changed to match. Originally, they had 3 numbers as in 000, 001, ..., 999, but due to differences between versions of the dataset, some folders had 4 numbers 0000, 0001, ..., 0999, so I just added a `trailing_zeros_folders.py` that takes care of all of that. More details in the respective repository.

# References

If you find this work useful, please consider citing:

* E. A. Rios, W.-H. Cheng, and B.-C. Lai, “DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset For Anime Character Recognition,” arXiv:2101.08674 [cs], Jan. 2021, Accessed: Jan. 22, 2021. [Online]. Available: http://arxiv.org/abs/2101.08674.
* Yan Wang, "Danbooru2018 Anime Character Recognition Dataset," July 2019. https://github.com/grapeot/Danbooru2018AnimeCharacterRecognitionDataset 
* Anonymous, The Danbooru Community, & Gwern Branwen; “Danbooru2020: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset”, 2020-01-12. Web. Accessed [DATE] https://www.gwern.net/Danbooru2020


```bibtex
@misc{rios2021dafre,
      title={DAF:re: A Challenging, Crowd-Sourced, Large-Scale, Long-Tailed Dataset For Anime Character Recognition}, 
      author={Edwin Arkel Rios and Wen-Huang Cheng and Bo-Cheng Lai},
      year={2021},
      eprint={2101.08674},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
    @misc{danboorucharacter,
        author = {Yan Wang},
        title = {Danbooru 2018 Anime Character Recognition Dataset},
        howpublished = {\url{https://github.com/grapeot/Danbooru2018AnimeCharacterRecognitionDataset}},
        url = {https://github.com/grapeot/Danbooru2018AnimeCharacterRecognitionDataset},
        type = {dataset},
        year = {2019},
        month = {July} }
```

```bibtex
    @misc{danbooru2020,
        author = {Anonymous and Danbooru community and Gwern Branwen},
        title = {Danbooru2020: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset},
        howpublished = {\url{https://www.gwern.net/Danbooru2020}},
        url = {https://www.gwern.net/Danbooru2020},
        type = {dataset},
        year = {2021},
        month = {January},
        timestamp = {2020-01-12},
        note = {Accessed: DATE} }
```


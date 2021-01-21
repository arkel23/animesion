Anime character recognition/classification using PyTorch.

### Features
* Variety of architectures to choose from: LeNet, Resnet18/152, ViT B-16/B-32/L-16/L-32
* Two datasets, moeImouto and DAF:re, with 173 and more than 3000 classes, respectively
* Pre-trained models for each of these architectures for image sizes of 128x128 or 224x224, trained for 50 or 200 epochs.
* Supporting scripts for visualization and stats of the datasets.
* Scripts for training from scratch, evaluation (accuracy of a model with a certain set and pretrained weights), and inference (classifies all images in test_images/ folder)

### To do
* Add support for continuing training from a checkpoint.
* Better documentation.
 


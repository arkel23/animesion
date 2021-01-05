#CUDA_VISIBLE_DEVICES=0 python train.py --name try_2 --dataset_name danbooruFaces --dataset_path "/edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 2 --batch_size 2048

#CUDA_VISIBLE_DEVICES=0 python train.py --name try_2 --dataset_name moeImouto --dataset_path "/edahome/pcslab/pcs05/edwin/data/moeimouto_animefacecharacterdataset/" --train_epochs 10 --batch_size 1024 --model_type resnet18 --pretrained True

#python train.py --name test --dataset_name moeImouto --dataset_path "/home2/yan/disk/edwin/personal/data/personal/moeimouto_animefacecharacterdataset/" --model_type B_16 --train_epochs 20 --pretrained True

CUDA_VISIBLE_DEVICES=0 python train.py --name try_3 --dataset_name moeImouto --dataset_path "/edahome/pcslab/pcs05/edwin/data/moeimouto_animefacecharacterdataset/" --train_epochs 10 --batch_size 16 --model_type H_14

#--pretrained False

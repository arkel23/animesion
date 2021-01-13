#CUDA_VISIBLE_DEVICES=0 python train.py --name try_2 --dataset_name danbooruFaces --dataset_path "/edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 2 --batch_size 2048

#CUDA_VISIBLE_DEVICES=0 python train.py --name try_2 --dataset_name moeImouto --dataset_path "/edahome/pcslab/pcs05/edwin/data/moeimouto_animefacecharacterdataset/" --train_epochs 10 --batch_size 1024 --model_type resnet18 --pretrained True

#python train.py --name test --dataset_name moeImouto --dataset_path "/home2/yan/disk/edwin/personal/data/personal/moeimouto_animefacecharacterdataset/" --model_type B_16 --train_epochs 20 --pretrained True

#CUDA_VISIBLE_DEVICES=0 python train.py --name try_3 --dataset_name moeImouto --dataset_path "/home2/yan/disk/edwin/personal/data/moeimouto_animefacecharacterdataset/" --train_epochs 10 --batch_size 32 --model_type L_16 --image_size 128

#CUDA_VISIBLE_DEVICES=0 python train.py --name danbooruFaces_resnet152_ptFalse_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type B_16 --image_size 128

#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --name danbooruFaces_b16_ptFalse_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type B_16 --image_size 128 > danbooruFaces_b16_ptFalse_batch64_imageSize128_50epochs_epochDecay20.txt &

#nohup python -u train.py --name danbooruFaces_l16_ptTrue_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type L_16 --image_size 128 --pretrained True > danbooruFaces_l16_ptTrue_batch64_imageSize128_50epochs_epochDecay20.txt

#nohup python -u train.py --name danbooruFaces_l16_ptFalse_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type L_16 --image_size 128 > danbooruFaces_l16_ptFalse_batch64_imageSize128_50epochs_epochDecay20.txt

#nohup python -u train.py --name danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type B_32 --image_size 128 --pretrained True > danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.txt

#nohup python -u train.py --name danbooruFaces_resnet18_ptTrue_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type resnet18 --image_size 128 --pretrained True > danbooruFaces_resnet18_ptTrue_batch64_imageSize128_50epochs_epochDecay20.txt

#nohup python -u train.py --name danbooruFaces_resnet152_ptTrue_batch64_imageSize128_50epochs_epochDecay20 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type resnet152 --image_size 128 --pretrained True > danbooruFaces_resnet152_ptTrue_batch64_imageSize128_50epochs_epochDecay20.txt

# nohup python -u train.py --name danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20_checkpointepoch50 --dataset_name danbooruFaces --dataset_path "/home2/yan/disk/edwin/personal/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type B_32 --image_size 128 --pretrained True --checkpoint_path "/home2/edwin_ed520/pytorch/projects/animesion/classification/checkpoints/danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.ckpt" > danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20_checkpointepoch50.txt

nohup python -u train.py --name moeImouto_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20_checkpointepoch50_transferfromdanbooru --dataset_name moeImouto --dataset_path "/home2/yan/disk/edwin/personal/data/moeimouto_animefacecharacterdataset/" --train_epochs 50 --epoch_decay 20 --batch_size 64 --model_type B_32 --image_size 128 --pretrained True --checkpoint_path "/home2/edwin_ed520/pytorch/projects/animesion/classification/checkpoints/danbooruFaces_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20.ckpt" --transfer_learning True > moeImouto_b32_ptTrue_batch64_imageSize128_50epochs_epochDecay20_checkpointepoch50_transferfromdanbooru.txt


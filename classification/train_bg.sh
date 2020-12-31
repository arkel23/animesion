#nohup python -u train.py danbooruFacesCrops > output.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --name test_run --train_epochs 10  > output.log &

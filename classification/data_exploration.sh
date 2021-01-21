#nohup python -u data_exploration.py --dataset_name danbooruFaces --dataset_path "/edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --split val --labels True --data_vis_full True > data_exploration_logs.out &

python data_exploration.py --dataset_name moeImouto --dataset_path "/edahome/pcslab/pcs05/edwin/data/moeimouto_animefacecharacterdataset/" --split val 

# --labels True --data_vis_full True

#python data_exploration.py --dataset_name danbooruFaces --dataset_path "/edahome/pcslab/pcs05/edwin/data/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/" --split val --image_size 32 --data_vis_partial True

#--labels True --data_vis_full True > data_exploration_logs.out &

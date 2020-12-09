import sys
from train.train import train

def main():
    dataset_name = sys.argv[1]
    print(dataset_name)
    #train(dataset_name='moeImouto')
    #train(dataset_name='danbooruFacesCrops')
    train(dataset_name = dataset_name)

if __name__ == '__main__':
    main()
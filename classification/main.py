import sys
import logging
import argparse

from train.train import train

logger = logging.getLogger(__name__)

def main():

    logging.basicConfig(filename='logs.txt', level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset_name", choices=["moeImouto", "danbooruFacesCrops"], 
                        default="moeImouto", help="Which dataset to use.")
    parser.add_argument("--model_type", choices=["shallow", 'resnet18', 'resnet152', 
                        'B_16', 'B_32', 'L_16', 'L_32', 'H_14',
                        'B_16_imagenet1k', 'B_32_imagenet1k', 
                        'L_16_imagenet1k', 'L_32_imagenet1k'],
                        default="shallow",
                        help="Which model architecture to use")
    parser.add_argument("--results_dir", default="results", type=str,
                        help="The directory where results will be stored")
    parser.add_argument("--image_size", default=128, type=int,
                        help="Image (square) resolution size")
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for train/val/test.")
    parser.add_argument("--train_epochs", default=200, type=int,
                        help="Total number of epochs for training.")                         
    parser.add_argument("--checkpoint_each_epochs", default=5, type=int,
                        help="Run prediction on validation set every so many epochs."
                        "Also saves checkpoints at this value."
                        "Will always run one test at the end of training.")
    parser.add_argument("--epoch_decay", default=100, type=int,
                        help="After how many epochs to decay the learning rate once.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Initial learning rate.")  
    parser.add_argument("--pretrained", choices=[True, False],
                        help="For models with pretrained weights available")                      
    args = parser.parse_args()

    logger.info(args)

    train(logger, args)            

if __name__ == '__main__':
    main()
import os, sys, glob
import pandas as pd
import numpy as np

def data_split(data_folder, split):
    # splits data into training and testing
    '''
    for each class count the number of instances
    then divide so training, val and test set contain 
    0.7, 0.1 and 0.2 numbers of data
    order by class_id
    then take 70% of values of each class (in order) 
    and make a new df from it
    save it (it should have roughly 70% of)
    repeat for val and test
    # comment on long-tailed cifar
    # the lt is only on the training set
    # test set has same amount of labels (600 each)
    # since model is only trained on training without knowing test
    # in the cifar-lt the test is going to be bad because model 
    # "overfits" to certain classes and gets high accuracy on them
    # but on others underrepresented classes performs badly
    # in my case the test accuracy should be higher since
    # ratio in test st is same as in training set
    # therefore looking at the per label accuracy is important 
    # to understand behavior of the lt classes
    # an alternative is to clip the testing set so that 
    # all classes have same amount of values as the one with less values
    # or maybe use the median/mean and if theres more than that just dont
    # use them
    '''
    '''
    alternative strategy for resampling: make it so that epochs no of value
    are roughly the same
    for example if one class has 6000 , others 600, and one with less 60
    make is to that training is in order of classes with less samples
    then when going to classes with more, only use values as many as the 
    one with less (60), and then jump to the next class 
    how to jump? since we know the labels, if the labels are of a class that
    has already reached its limit, then break;
    that way it will keep breaking on that class until it reaches the next class
    '''

def main():
    data_folder = os.path.abspath(sys.argv[1])
    split_train = sys.argv[2]
    split_val = sys.argv[3]
    split_test = sys.argv[4]
    if (split_train+split_val+split_test == 1):
        split = [split_train, split_val, split_test]
    else:
        split = [0.7, 0.1, 0.2]
    data_split(data_folder, split)
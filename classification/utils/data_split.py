import os, sys, glob
import pandas as pd
import numpy as np

def data_split(data_dic_path, split):
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
    
    df = pd.read_csv(data_dic_path, sep=',', names=['class_id', 'dir'])
    print('Original df: ', len(df))
    
    samples_per_class_df = df.groupby('class_id', as_index=True).count()
    
    df_list_train = []
    df_list_test = []
    for class_id, total_samples_class in enumerate(samples_per_class_df['dir']):
        train_samples_class = int(total_samples_class*split[0])
        test_samples_class = total_samples_class - train_samples_class
        assert(train_samples_class+test_samples_class==total_samples_class)
        train_subset_class = df.loc[df['class_id']==class_id].groupby('class_id').head(train_samples_class)
        test_subset_class = df.loc[df['class_id']==class_id].groupby('class_id').tail(test_samples_class)
        df_list_train.append(train_subset_class)
        df_list_test.append(test_subset_class)
    
    df_train = pd.concat(df_list_train)
    df_test = pd.concat(df_list_test)

    print('Train df: ')
    print(df_train.head())
    print(df_train.shape)
    print('Test df: ')
    print(df_test.head())
    print(df_test.shape)

    dataset_name = os.path.basename(os.path.normpath(data_dic_path))
    df_train_name = dataset_name + '_train.csv'
    df_train.to_csv(df_train_name, sep=',', header=False, index=False)

    df_test_name = dataset_name + '_test.csv'
    df_test.to_csv(df_test_name, sep=',', header=False, index=False)
    print('Finished saving train and test split dictionaries.')
    

def main():
    data_dic_path = os.path.abspath(sys.argv[1])
    try:
        split_train = float(sys.argv[2])   # % of data for train (def: 0.8)
        split_test = float(sys.argv[3])     # % of data for val (def: 0.2)
        assert split_train+split_test==1, 'Arguments for split ratios should add up to 1'
        split = [split_train, split_test]
    except:
        split = [0.8, 0.2]
    
    print(split)
    data_split(data_dic_path, split)

main()
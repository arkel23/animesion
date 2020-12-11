import os, sys, glob
import pandas as pd
import numpy as np

def make_data_dic(data_folder):
    # makes an imagefolder (imagenet style) with images of class in a certain folder
    # into a txt dictionary with the first column being the file dir (relative)
    # and the second into the class
    types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(data_folder, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))    
    print('Total image files', len(files_all))

    class_name_list = []      # holds classes names and is also relative path
    filename_classid_dic = {} #  filename and classid pairs
    class_id_name_dic = {}    # id and class name/rel path as dict
    
    idx = -1
    for file_path in files_all:
        abs_path, filename = os.path.split(file_path)
        _, class_name = os.path.split(abs_path)
        rel_path = os.path.join(class_name, filename)

        if class_name not in class_name_list:
            idx += 1
            class_name_list.append(class_name)
            class_id_name_dic[idx] = class_name 

        filename_classid_dic[rel_path] = idx
            
    
    df = pd.DataFrame.from_dict(filename_classid_dic, orient='index', columns=['class_id'])
    idx_col = np.arange(0, len(df), 1)
    df['idx_col'] = idx_col
    df['file_rel_path'] = df.index
    df.set_index('idx_col', inplace=True)
    print(df.head())    
            
    no_classes = idx
    print('Total number of classes: ', no_classes)

    dataset_name = os.path.basename(os.path.normpath(data_folder))
    df_name = dataset_name + '.csv'
    df.to_csv(df_name, sep=',', header=False, index=False)

def main():
    # input is the path to the folder with imagenet-like structure
    # imagenet/
    # imagenet/class1/
    # imagenet/class2/
    # ...
    # imagenet/classN/
    data_folder = os.path.abspath(sys.argv[1])
    make_data_dic(data_folder)

main()


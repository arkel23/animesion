import os, sys, glob
import cv2
import numpy as np

def data_count():
    # counts no of files of a (or multiple) type(s) in a directory
    # for root, dirs, files in os.walk(directory)
    pass

def data_vis():
    # shows a pic in a directory
    pass

def data_split():
    # splits data into training and testing
    pass

def data_dict_hua():
    # makes an imagefolder (imagenet style) with images of class in a certain folder
    # into a txt dictionary with the first column being the file dir (relative)
    # and the second into the class
    pass

def main():
    fields = {'data_count': data_vis, 'data_vis': data_vis,
    'data_split': data_split, 'data_dict_hua': data_dict_hua}
    print(fields.keys())
    
    func = input('Input function to use: ')
    fields[func]
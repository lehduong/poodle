import os
import numpy as np
import argparse
from os import listdir
from os.path import isfile, isdir, join
import random
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ISIC/', type=str, help='path to the data')
    parser.add_argument('--split', default='data/split/isic/', type=str, help='path to the split folder')
    args = parser.parse_args()
    dataset_list = ['train','val','test']
    #
    img_path = join(args.data, 'ISIC2018_Task3_Training_Input')
    csv_path = join(args.data, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')

    # Read the csv file
    data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

    # First column contains the image paths
    image_name = np.asarray(data_info.iloc[:, 0])

    labels = np.asarray(data_info.iloc[:, 1:])
    labels = (labels != 0).argmax(axis=1)

    # Calculate len
    data_len = len(image_name)

    classfile_list_all = [[],[],[],[],[],[],[]]
    for i in range(data_len):
        #print(i, len(labels), len(image_name))
        classfile_list_all[labels[i]].append(image_name[i]+'.jpg')

    if not os.path.isdir(args.split):
        os.makedirs(args.split)
        
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        l = int(len(classfile_list)/100*20)
        train_classfile_list = classfile_list[:l]
        test_classfile_list = classfile_list[l:]
        if 'train' in dataset:
            file_list = file_list + train_classfile_list
            label_list = label_list + np.repeat(i, len(train_classfile_list)).tolist()

        if 'test' in dataset:
            file_list = file_list + test_classfile_list
            label_list = label_list + np.repeat(i, len(test_classfile_list)).tolist()

    if 'train' in dataset:
        with open(join(args.split + 'train.csv'), 'w') as f1:
            f1.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])
    #if 'val' in dataset:
    #    with open('./data/isic' + '/val.csv', 'w') as f2:
    #        f2.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])
    if 'test' in dataset:
        with open(join(args.split + 'test.csv'), 'w') as f3:
            f3.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])

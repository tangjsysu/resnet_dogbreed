import torch
import os
import shutil
import pandas as pd
import random
# set random seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

data_dir = '../dog-breed-identification'  # the dataset path
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'
new_data_dir = '../train_valid_test'
valid_ratio = 0.1  # the percentage of validation data


def mkdir_if_not_exist(path):
    # if the directory does not exist, create it
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio):
    # read the labels
    labels = pd.read_csv(os.path.join(data_dir, label_file))
    id2label = {Id: label for Id, label in labels.values}  # (key: value): (id: label)

    # ramdomly shuffle the data
    train_files = os.listdir(os.path.join(data_dir, train_dir))
    random.shuffle(train_files)

    # organize the dataset
    valid_ds_size = int(len(train_files) * valid_ratio) # number of validation samples
    for i, file in enumerate(train_files):
        img_id = file.split('.')[0]
        img_label = id2label[img_id]
        if i < valid_ds_size:
            mkdir_if_not_exist([new_data_dir, 'valid', img_label])
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'valid', img_label))
        else:
            mkdir_if_not_exist([new_data_dir, 'train', img_label])
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'train', img_label))
        mkdir_if_not_exist([new_data_dir, 'train_valid', img_label])
        shutil.copy(os.path.join(data_dir, train_dir, file),
                    os.path.join(new_data_dir, 'train_valid', img_label))

    # test set
    mkdir_if_not_exist([new_data_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(new_data_dir, 'test', 'unknown'))


reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio)
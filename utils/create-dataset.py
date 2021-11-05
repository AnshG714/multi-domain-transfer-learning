#!/usr/bin/env python
# coding: utf-8

import scipy.io
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# np.set_printoptions(threshold=np.inf)

DATA_PATH_PREFIX = "./datasets" #assuming we're running this from the project root.

# load the flowers dataset -- [https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/]
labels = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'flowers/imagelabels.mat'))["labels"].flatten()
split = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'flowers/setid.mat'))

# make the splits - each of these is an array (for some reason, the test and train labels have been split)
train_split = split["tstid"].flatten()
test_split = split["trnid"].flatten()
val_split = split["valid"].flatten()

def prepare_flowers_df(split):
    """
    split: A numpy array containing a list of image ids for this split
    
    Returns: A Pandas DataFrame containing 2 columns: img_name, which contains the image name in the form image_xxxxx.jpg, and a label signifying which flower it is.
    """
    
    def image_name(n):
        s = str(n)
        return "0" * (5 - len(s)) + s + ".jpg"
    
    df = pd.DataFrame(columns=["img_name", "label"])
    df["img_name"] = np.array([image_name(n) for n in split])
    df["label"] = np.array([labels[n - 1] for n in split]) - 1
    return df

train_df = prepare_flowers_df(train_split)
test_df = prepare_flowers_df(test_split)
val_df = prepare_flowers_df(val_split)

train_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'flowers/train_csv.csv'), index = False, header=True)
test_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'flowers/test_csv.csv'), index = False, header=True)
val_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'flowers/val_csv.csv'), index = False, header=True)

# print("Training set size for flowers", train_df.shape)
# print("Testing set size for flowers", test_df.shape)
# print("Validation set size for flowers", val_df.shape)

# Now, we handle the cars dataset -- [https://ai.stanford.edu/~jkrause/cars/car_dataset.html]
# NOTE: Both the test and validation set use the cars_test image directory
train_split = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'cars/devkit/cars_train_annos.mat'))["annotations"][0]
res = []
for el in list(train_split):
    res.append(np.array(list(el)))
train_split = np.array(res).reshape((-1, 6))

test_and_val_split = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'cars/devkit/cars_test_annos_withlabels.mat'))["annotations"][0]
res = []
for el in list(test_and_val_split):
    res.append(np.array(list(el)))
test_and_val_split = np.array(res).reshape((-1, 6))
test_split, val_split = train_test_split(test_and_val_split, test_size = 0.5, random_state = 42)

def prepare_cars_df(split):
    """
    split: In this case, it's a 2d numpy array who's second to last column is the label and the last column is the image name
    """
    df = pd.DataFrame(columns=["img_name", "label"])
    df["img_name"] = split[:, -1]
    df["label"] = split[:, -2] - 1
    return df

train_df = prepare_cars_df(train_split)
test_df = prepare_cars_df(test_split)
val_df = prepare_cars_df(val_split)

train_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'cars/train_csv.csv'), index = False, header=True)
test_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'cars/test_csv.csv'), index = False, header=True)
val_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'cars/val_csv.csv'), index = False, header=True)

# print("Training set size for cars", train_df.shape)
# print("Testing set size for cars", test_df.shape)
# print("Validation set size for cars", val_df.shape)

# Prepare dogs dataset

def prepare_dogs_df(mat_contents):
    files = []
    for el in list(mat_contents["file_list"]):
        files.append(np.array(list(el[0])))
    files = np.array(files).flatten()

    labels = []
    for el in list(mat_contents["labels"]):
        labels.append(np.array(list(el)))
    labels = np.array(labels).flatten()

    df = pd.DataFrame(columns = ["img_name", "label"])
    df["img_name"] = files
    df["label"] = labels - 1

    return df

test_list_mat = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'dogs/test_list.mat'))
train_list_mat = scipy.io.loadmat(os.path.join(DATA_PATH_PREFIX, 'dogs/train_list.mat'))
test_df = prepare_dogs_df(test_list_mat)
train_df = prepare_dogs_df(train_list_mat)

# split the test_df to test and val sets
test_df, val_df = train_test_split(test_df, test_size = 0.5, random_state = 42)

train_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'dogs/train_csv.csv'), index = False, header=True)
test_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'dogs/test_csv.csv'), index = False, header=True)
val_df.to_csv(os.path.join(DATA_PATH_PREFIX, 'dogs/val_csv.csv'), index = False, header=True)





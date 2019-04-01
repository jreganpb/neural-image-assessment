import numpy as np
import os
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

# path to the images and the text file which holds the scores and ids
base_images_path = r'/mnt/ds3rdparty/AVA/data/images/'
base_data_path = r'/mnt/ds3rdparty/AVA/style_image_lists/'
base_disk_path = r'/home/ubuntu/AVA/images/'
ava_dataset_path = r'/mnt/ds3rdparty/AVA/AVA.txt'

IMAGE_SIZE = 224 # Keras accepts None for height and width fields.

def get_available_files_s3(pathname,bucket='ds3rdparty',callsystem=False):
    if callsystem:
        os.system('rm /home/ubuntu/data.txt')
        q = 'sudo aws s3 ls s3://' + bucket + '/' + pathname + '/ > /home/ubuntu/data.txt'
        os.system(q)
    with open('/home/ubuntu/data.txt', 'r') as f:
        dat = f.readlines()
    dat = sorted(dat)
    iidnums = {}
    files = []
    for i in dat:
        tmp = i.rstrip().split()
        iid = int(tmp[3].split('.')[0])
        iidnums[iid] = 1
        files.append(base_images_path + tmp[3])
    return iidnums, sorted(files)

def get_available_files_disk(pathname='/home/ubuntu/AVA/images/'):
    l = os.listdir(pathname)
    iidnums = {}
    files = []
    for i in l:
        iid = int(i[0].split('.')[0])
        iidnums[iid] = 1
        files.append(pathname + i)
    return iidnums, files

def pad_image(image):
    height, width, dim = image.shape
    diff = int(float(abs(width-height)) / 2.)
    top = 0; bottom = 0; left = 0; right = 0
    if height >= width: # Pad wide
        left = diff; right = diff
    elif width >= height: # Pad height
        top = diff; bottom = diff
    image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
    return image

def parse_data_without_augmentation(filename):
    '''
    Loads the image file without any augmentation. Used for validation set.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = cv2.imread(filename)
    image = pad_image(image)
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
    image = image / 255.
    return image

def image_generator(files,scores,batch_size=64):
    indexes = np.array(range(len(files)))
    while True:
        paths = np.random.choice(a=indexes,size=batch_size)
        batch_input = []; batch_output = []; batch_weight = []
        for ix in paths:
            #img = cv2.imread(filename)
            filename = files[ix]; yvar=scores[ix]
            try:
                inp = parse_data_without_augmentation(filename)
            except Exception:
                continue
            batch_input.append(inp); batch_output.append(yvar)
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield(batch_x,batch_y)

def calculating_class_weights(y_true:np.ndarray):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = y_true.shape[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights

with open(base_data_path + 'train.jpgl','r') as f:
    style_images = f.readlines()
    for d in range(len(style_images)):
        style_images[d] = int(style_images[d].rstrip())

with open(base_data_path + 'train.lab','r') as f:
    style_labels = f.readlines()
    for d in range(len(style_labels)):
        style_labels[d] = int(style_labels[d].rstrip())

train_files = []; train_labels = []
all_files = os.listdir(base_disk_path)

for ix, i in enumerate(style_images):
    if str(i) + '.jpg' not in all_files:
        print('%d not found in directory, continuing...' % (i))
        continue
    fname = base_disk_path + '/' + str(i) + '.jpg'
    train_files.append(fname); train_labels.append(style_labels[ix])

enc = OneHotEncoder()
train_wts = calculating_class_weights(np.array(train_labels))
train_files = np.array(train_files); train_labels = enc.fit_transform(np.array(train_labels).reshape(len(train_labels),1)).toarray()

with open(base_data_path + 'test.jpgl','r') as f:
    tmp_images = f.readlines()
    for d in range(len(tmp_images)):
        tmp_images[d] = int(tmp_images[d].rstrip())

with open(base_data_path + 'test.multilab','r') as f:
    tmp = f.readlines()
    for d in range(len(tmp)):
        tmp[d] = [int(x) for x in tmp[d].rstrip().split(' ')]

test_files = []; test_labels = []
for ix, i in enumerate(tmp_images):
    if str(i) + '.jpg' not in all_files:
        print('%d not found in directory, continuing...' % (i))
        continue
    fname = base_disk_path + '/' + str(i) + '.jpg'
    test_files.append(fname); test_labels.append(tmp[ix])

test_files = np.array(test_files); test_labels = np.array(test_labels)

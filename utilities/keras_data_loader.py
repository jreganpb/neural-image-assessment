import numpy as np
import os
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# path to the images and the text file which holds the scores and ids
base_images_path = r'/mnt/s3/AVA/data/images/'
ava_dataset_path = r'/mnt/s3/AVA/AVA.txt'

IMAGE_SIZE = 256 # Keras accepts None for height and width fields.

def get_available_files(pathname,bucket='ds3rdparty'):
    os.system('rm /tmp/data.txt')
    q = 'sudo aws s3 ls s3://' + bucket + '/' + pathname + '/ > /tmp/data.txt'
    os.system(q)
    with open('/tmp/data.txt', 'r') as f:
        dat = f.readlines()
    dat = sorted(dat)
    iidnums = {}
    files = []
    for i in dat:
        tmp = i.rstrip().split()
        iid = int(tmp[3].split('.')[0])
        iidnums[iid] = 1
        files.append(base_images_path + i)
    return iidnums, sorted(files)

#files = glob.glob(base_images_path + "*.jpg")
#files = sorted(files)

iidnums, files = get_available_files(pathname='/'.join(base_images_path.split('/')[3:6]))

train_image_paths = []
train_scores = []
train_files = []

print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    i2 = 0
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])
        if id not in iidnums: ## File isn't on S3, continue...
            continue
        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(id) + '.jpg'
        filename = str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)
            train_files.append(filename)

        count = len(files) // 20
        if i2 % count == 0 and i2 != 0:
            print('Loaded %d percent of the dataset' % (i2 / float(len(files)) * 100))
        i2 = i2 + 1

train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

val_image_paths = train_image_paths[-5000:]
val_scores = train_scores[-5000:]
val_files = train_scores[-5000:]
train_image_paths = train_image_paths[:-5000]
train_scores = train_scores[:-5000]
train_files = train_files[:-5000]

train_df = pd.DataFrame({'filename' : train_files})
train_df = pd.concat([train_df,pd.DataFrame(train_scores)],axis=1,ignore_index=True).reset_index(drop=True)
train_df.columns = ['filename','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']

val_df = pd.DataFrame({'filename' : val_files})
val_df = pd.concat([val_df,pd.DataFrame(val_scores)],axis=1,ignore_index=True).reset_index(drop=True)
val_df.columns = train_df.columns

print('Train set size : ', train_image_paths.shape, train_scores.shape)
print('Val set size : ', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready !')

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


#datagen = ImageDataGenerator(preprocessing_function=parse_data_without_augmentation)
datagen = ImageDataGenerator(preprocessing_function=parse_data_without_augmentation)

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                              directory=base_images_path,
                                              x_col='filename',
                                              y_col=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'],
                                              batch_size=32)

val_generator = datagen.flow_from_dataframe(dataframe=val_df,
                                            directory=base_images_path,
                                            x_col='filename',
                                            y_col=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'],
                                            batch_size=32)
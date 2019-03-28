import numpy as np
import os
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import gc

# path to the images and the text file which holds the scores and ids
base_images_path = r'/mnt/ds3rdparty/AVA/data/images/'
#base_images_path = r'/home/ubuntu/AVA/images/'
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

#files = glob.glob(base_images_path + "*.jpg")
#files = sorted(files)

#iidnums, files = get_available_files_s3(pathname='/'.join(base_images_path.split('/')[3:6]))
iidnums, files = get_available_files_disk()

train_image_paths = [None] * len(files)
train_scores = [None] * len(files)
train_files = [None] * len(files)
train_weights = [None] * len(files)
scores = {}; weights = {} # use weights for images based on votes; average photo in AVA had 210 votes

gc.disable()
print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        iid = int(token[1])
        values = np.array(token[2:12], dtype='float32')
        weights[iid] = np.sqrt(float(values.sum()) / 210.) # Use sqrt since 400 votes is only 2x as good as 100, etc
        values /= values.sum()
        scores[iid] = values

count = len(files) // 20
idx2 = 0

for idx, f in enumerate(files):
    f2 = f.split('/')
    fname = f2[len(f2)-1]
    fsplit = fname.split('.')
    iid = int(fsplit[0])
    if iid not in scores:
        continue
    train_image_paths[idx2] = f
    train_files[idx2] = fname
    train_scores[idx2] = scores[iid]
    train_weights[idx2] = weights[iid]
    idx2 = idx2 + 1

    if idx % count == 0 and idx != 0:
        print('Loaded %d percent of the dataset' % (idx / float(len(files)) * 100))

gc.enable()
gc.collect()
train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')
train_weights = np.array(train_weights)

val_image_paths = train_image_paths[-5000:]
val_scores = train_scores[-5000:]
val_files = train_files[-5000:]
val_weights = train_weights[-5000:]
train_image_paths = train_image_paths[:-5000]
train_scores = train_scores[:-5000]
train_files = train_files[:-5000]
train_weights = train_weights[:-5000]

train_y = {}; val_y = {}; train_wts = {}; val_wts = {}
for f in val_files:
    f2 = f.split('.')
    iid = int(f2[0])
    val_y[iid] = scores[iid]
    val_wts[iid] = weights[iid]

for f in train_files:
    f2 = f.split('.')
    iid = int(f2[0])
    train_y[iid] = scores[iid]
    train_wts[iid] = weights[iid]

train_df = pd.DataFrame({'filename' : train_image_paths})
train_df = pd.concat([train_df,pd.DataFrame(train_scores)],axis=1,ignore_index=True).reset_index(drop=True)
train_df.columns = ['filename','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']

val_df = pd.DataFrame({'filename' : val_image_paths})
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

def image_generator(files,scores,weights=None,batch_size=64):
    if weights is None: # Weights aren't being passed, so weigh everything as 1
        weights = {}
    while True:
        paths = np.random.choice(a=files,size=batch_size)
        batch_input = []; batch_output = []; batch_weight = []
        for filename in paths:
            #img = cv2.imread(filename)
            f2 = filename.split('/')
            fname = f2[len(f2) - 1]
            fsplit = fname.split('.')
            yvar = scores[int(fsplit[0])]
            wt = weights.get(int(fsplit[0]),1)
            try:
                inp = parse_data_without_augmentation(filename)
            except Exception:
                continue
            batch_input.append(inp); batch_output.append(yvar); batch_weight.append(wt)
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        batch_wt = np.array(batch_weight)
        yield(batch_x,batch_y,batch_wt)
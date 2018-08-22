#!/usr/bin/env python

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

import os
import cv2
import logging

from skimage.io import imread
import matplotlib.pyplot as plt
import sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..'))  # add the current directory
    import lime
from lime import lime_image
import time
from pycocotools.coco import COCO

from coco_preprocessing import *
from expl_train_utils import *
from ray.rllib.utils.timer import TimerStat

from keras_contrib.applications.resnet import ResNet18

# ### Load Dataset

# In[9]:

dataDir = ''
dataType = 'val2017'
annFile = 'annotations/instances_{}.json'.format(dataType)

num_samples = 100
batch_size = 10
num_classes = 10
NUM_CATEGORIES = 10
num_batches = num_samples // batch_size

start_iter = 0
num_training_samples = 100
num_iterations = 1
samples_per_iter = num_training_samples // num_iterations
explainer_batch_size = 128
batch_size = 128

# Get categories with largest number of images to use
def get_largest_categories(coco, num_categories):
    """
    Returns:
        list of {'supercategory': 'person', 'id': 1, 'name': 'person'}"""
    img_per_category = [(i, len(coco.getImgIds(catIds=i))) for i in range(90)]
    img_per_category.sort(key=lambda x: x[1])
    usedCatImgs = img_per_category[-num_categories:]  # list of (catId, numImgs) tuples

    # number of images available for smallest used cat(egory)
    minNumImgs = usedCatImgs[0][1]
    used_ids = [tup[0] for tup in usedCatImgs]  # list of catIds used

    used_categories = coco.loadCats(coco.getCatIds(catIds=used_ids))
    cat_names = [cat['name'] for cat in used_categories]
    logger.info('{} COCO categories used: \n{}\n'.format(
        len(used_categories), ' '.join(cat_names)))
    return used_categories




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    timers = {k: TimerStat() for k in ["predict", "preprocess"]}
    # initialize COCO api for instance annotations
    logger.info("Initialize Coco.")
    coco_instance = COCO(annFile)
    
    largest_categories = get_largest_categories(coco_instance, NUM_CATEGORIES)
    
    catId_to_catName = {d['id']: d['name'] for d in largest_categories}
    used_ids = list(catId_to_catName)

    # ### Load All Images
    imgIds = sum([coco_instance.getImgIds(catIds=used_id) for used_id in used_ids], [])

    data = []
    for img_info in coco_instance.loadImgs(imgIds)[:100]:
        with timers["preprocess"]:
            data += [preprocess(coco_instance, img_info, used_ids)]

    logger.info("Average Preprocess Time: {} seconds".format(timers["preprocess"].mean))
    data = np.array(data)
    np.random.shuffle(data)
    logger.info("Finished loading data.")

    model = ResNet18((96, 96, 3), 10)
    model.compile(
        optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # In[20]:

    x_train = data[:, 0]  # Bbox images

    # bbox_imgs = []
    temp = []
    for bbox_imgs in x_train:
        for img in bbox_imgs:
            if len(img.shape) != 3:
                img = np.stack((img, ) * 3, -1)
    #         bbox_imgs.append(img)
            alt_img = cv2.resize(
                img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
            temp.append(alt_img)

    x_train = np.array(temp)


    y_train = data[:, 1]
    y_train = np.array([label[0] for labels in y_train for label in labels])

    temp = np.zeros((y_train.shape[0], NUM_CATEGORIES))
    for ft in used_ids:
        vec = np.zeros(NUM_CATEGORIES)
        vec[used_ids.index(ft)] = 1
        temp[np.where(y_train == ft)] = vec

    y_train = temp

    # In[59]:

    configs = np.array([config for configs in data[:, 2] for config in configs])

    # In[24]:
    shapely_polygons = np.array(
        [polygon for polygons in data[:, 3] for polygon in polygons])

    # In[25]:

    # Single Image Inference Time
    times = []
    for i in range(100):
        with timers["predict"]:
            preds = model.predict(x_train[i][np.newaxis, ])
    logger.info("Single Image Inference: {} seconds".format(timers["predict"].mean))

    model.load_weights('saved_weights/15.hdf5')

    # In[83]:

    # # Using Epsilon-Greedy
    # sp = SubPlotter(num_iterations, use_bandits=False)

    # for iteration in range(start_iter, start_iter + num_iterations):
    #     start = iteration * samples_per_iter
    #     end = (iteration + 1) * samples_per_iter

    #     x_train_batch = x_train[start:end]
    #     y_train_batch = y_train[start:end]
    #     configs_batch = configs[start:end]
    #     shapely_polygons_batch = shapely_polygons[start:end]
    #     sp.plot_rand_segs(
    #         model,
    #         used_ids,
    #         catId_to_catName,
    #         x_train_batch,
    #         y_train_batch,
    #         shapely_polygons_batch,
    #         configs_batch,
    #         num_segs=10)

    #     print("loss, accuracy = {}\n".format(
    #         model.evaluate(x_train, y_train, batch_size=128, verbose=0)))

    # sp.end()

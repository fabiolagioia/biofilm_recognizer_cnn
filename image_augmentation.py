# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:04:30 2019

@author: Fabio La Gioia
"""

import os
import cv2
from skimage import io
import numpy as np
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator

def imgGen(img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,  preprocess_fcn=None, batch_size=20):
    """
    Datagen generation, tool to carry out the various image 
    augmentation transformations automatically, based on the 
    value of the parameters set. The DataGen is returned.
    Parameters:
    ----------
    - img, zca, rotation, w_shift, h_shift, shear, zoom, h_flip,
      vflip, preprocess_fcn, batch size
    #Return:
    ----------
    - datagen
    """
    datagen = ImageDataGenerator(
            zca_whitening = zca,
            rotation_range = rotation,
            width_shift_range = w_shift,
            height_shift_range = h_shift,
            shear_range = shear,
            zoom_range = zoom,
            fill_mode = 'nearest',
            horizontal_flip = h_flip,
            vertical_flip = v_flip,
            preprocessing_function = preprocess_fcn,
            data_format = K.image_data_format())
    
    datagen.fit(img)
    return datagen
 
def image_augmentation(path):
    """
    Loading images from the directory defined by the path.
    This method performs the image augmentation.
    Parameters:
    ----------
    - path: directory path
    """
    for file in os.listdir(path):
        image = cv2.cvtColor(cv2.imread(path + "/" + file), cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        image /= 255
        h_dim = np.shape(image)[0]
        w_dim = np.shape(image)[1]
        num_channel = np.shape(image)[2]
        image = image.reshape(1, h_dim, w_dim, num_channel)
        dataGen = imgGen(image, rotation = 30, h_shift = 0.3)
        i = 0
        for img_batch in dataGen.flow(image, batch_size = 20, shuffle = False):
            for img in img_batch:
                io.imsave(path + str(i) + "-" + file, img)
                i = i  +1    
            if i >= 20:
                break
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:10:33 2019

@author: Fabio La Gioia
"""
import os
import cv2
from skimage import io

def create_dataset(path):
    """
    Loading images from the directory defined by the path.
    This method performs the image segmentation.
    Parameters:
    ----------
    - path: directory path
    """
    for file in os.listdir(path):
        image = cv2.cvtColor(cv2.imread(path + "/" + file), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        step = 384
        height = height - (height % step)
        width = width - (width % step)
        c = 0
        for y in range(0, height, step):
            for x in range(0, width, step):
                crop = image[y : y+step, x : x + step]
                io.imsave(os.getcwd() + "/tiles/" + str(c) + "-" + file, crop)
                c+=1

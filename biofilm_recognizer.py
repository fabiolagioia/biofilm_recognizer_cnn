# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:35:13 2019

@author: Fabio La Gioia
"""

import numpy as np
import os
import cv2
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, flatten, fully_connected
from tflearn.layers.estimator import regression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 


DIRECTORY = os.getcwd() + '/dataset'
IMG_SIZE = 50
LearningRate = 1e-3

MODEL_NAME = 'Biofilm-{}-{}.model'.format(LearningRate, '2conv-basic') # just so we remember which saved model is which, sizes must match

def unify_set(x,y):
    data = []
    for el_x, el_y  in zip(x, y):
        data.append([el_x, el_y])
    return data

def create_data():
    """
    In this method, each image is converted to grayscale and
    resized.
    This method returns two arrays, one containing the images and
    one containing the label. The label shown corresponds to the
    name of the folder ("Biofilm" or "Other") containing the
    image.
    Parameters:
    -----------
    Return:
    -----------
    - x: image array
    - y: label array
    """
    dataset = []
    for cl in os.listdir(DIRECTORY):
        if cl != "Unlabelled":
            for img in os.listdir(os.path.join(DIRECTORY, cl)):
                if cl == "Biofilm":
                    label = [1,0]
                elif cl == "Other":
                    label = [0,1]
                path = os.path.join(os.path.join(DIRECTORY, cl), img)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                dataset.append([np.array(img), np.array(label)])
    shuffle(dataset)
    x = [] 
    y = []
    for el in dataset:
        x.append(el[0])
        y.append(el[1])


    return x, y

def data_split(x, y, train_index, validation_index):
    """
    This method allows to divide the data set into x_train,
    x_test. y_train and y_test;
    Parameters:
    ----------
    - x, y, train_index, validation_index
    Return:
    ----------
    - x_train, y_train, x_test, y_test
    """
    x_train = []
    y_train = []    
    x_test = []
    y_test = []
    for i in train_index:
        x_train.append(x[i])
        y_train.append(y[i])
    for i in validation_index:
        x_test.append(x[i])
        y_test.append(y[i])
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test
    
def conf_matrix(testLabel,test):
    """
    In this method a confusion matrix is calculated with the ai
    of verifying how many tiles were incorrectly classified.
    Parameters:
    -----------
    - testLabel, test
    """
    conf_mat = confusion_matrix(testLabel,test)
    f, ax = plt.subplots(figsize = (3, 3))
    sns.heatmap(conf_mat, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.show()
    
def create_net():
    """
    In this method che convolutional neural network is created.
    Return:
    ---------
    - model
    """
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name = 'input')

    convnet = conv_2d(convnet, 32, 5, activation = 'relu')
    convnet = max_pool_2d(convnet, 5)
    
    
    convnet = conv_2d(convnet, 64, 5, activation = 'relu')
    convnet = max_pool_2d(convnet, 5)
    
    
    convnet = conv_2d(convnet, 128, 5, activation = 'relu')
    convnet = max_pool_2d(convnet, 5)
    
    
    convnet = conv_2d(convnet, 64, 5, activation = 'relu')
    convnet = max_pool_2d(convnet, 5)
    
    
    convnet = conv_2d(convnet, 32, 5, activation = 'relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = flatten(convnet)
    
    convnet = fully_connected(convnet, 1024, activation = 'relu')
    convnet = dropout(convnet, 0.8)
   
    convnet = fully_connected(convnet, 2, activation = 'softmax')
    convnet = regression(convnet, optimizer = 'adam', learning_rate = LearningRate,
                         loss = 'categorical_crossentropy', name = 'targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir = 'log')
    return model
   
def evaluate_classifier(x_test, y_test, model):
    """
    This method allows to evaluate the trained model, in terms of
    accuracy, sensitivity and miss rate based on the test set;
    Parameters:
    -----------
    - x_test, y_test, model
    """
    test = model.predict(x_test)        
    testLabel = []
    test_value = []
    for y_el, el in zip(y_test, test):
        testLabel.append(np.argmax(y_el))
        test_value.append(np.argmax(el))
    print(classification_report(testLabel, test_value, target_names  = ['Biofilm', 'Other']))
    conf_matrix(testLabel, test_value)

def fit_model (x,y):
    """
    Method in which the training of the network is carried out,
    the best model is saved, obtained after the application of k-
    fold cross-validation, and the accuracy of the entire system
    is calculated. While, if the training was carried out
    previously, the model is loaded;
    Parameters:
    -----------
    - x, y
    Return:
    -----------
    - model
    """
    k = 10
    if os.path.exists(os.path.join("Checkpoint_CNN", "{}.meta".format(MODEL_NAME))):
        model = create_net()
        model.load(os.path.join("Checkpoint_CNN", MODEL_NAME))
        print('model loaded!') 
        return model
    else:
        models = []
        best_acc = 0
        kf = KFold(n_splits = k)
        i = 0
        tot_acc= []
        for train_index, validation_index in kf.split(x, y):
            print("VALIDATION: ")
            print(i+1)
            x_train, y_train, x_test, y_test = data_split(x, y, train_index, validation_index)
            model = create_net()
            model.fit({'input': x_train}, 
                  {'targets': y_train}, 
                  n_epoch = 70, 
                  validation_set = ({'input': x_test}, 
                                  {'targets':y_test}),
                  snapshot_step = 500, 
                  show_metric = True, 
                  run_id = MODEL_NAME)
            val_acc = model.evaluate(x_test, y_test)
            tot_acc.append(val_acc)
            if(val_acc[0] > best_acc):
                index_best = i
                best_acc = val_acc
                x_test_best = x_test
                y_test_best = y_test
            models.append(model)
            i = i + 1
        models[index_best].save(os.path.join("Checkpoint_CNN", MODEL_NAME))
        print('Estimated Accuracy with K-Fold %.3f' % (np.mean(tot_acc)))
        print("The best CNN selected is:")
        evaluate_classifier(x_test_best, y_test_best, models[index_best])
        print("Modello migliore: ")
        print(models[index_best])
        print(index_best)
        return models[index_best]

def predict():
    """
    Method in which the prediction is made for all the images
    contained in the "Unlabeled" folder. This method prints all
    the predictions made;
    """
    fig=plt.figure(figsize=(5, 40))
    path_unlabelled = os.path.join(DIRECTORY, "Unlabelled")
    list = os.listdir(path_unlabelled)
    number_files = len(list)
    if (number_files % 4 == 0):
        num_rows = number_files / 4
    else:
        num_rows = int((number_files / 4) + 1)
    i=1
    for img in os.listdir(path_unlabelled):
        y = fig.add_subplot(num_rows, 4, i)
        i +=1
        orig = cv2.imread(os.path.join(path_unlabelled, img))
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img = cv2.imread(os.path.join(path_unlabelled,img), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data = np.array(img).reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 0: 
            str_label = 'Biofilm'
        else: 
            str_label = 'Other'
    
        y.imshow(orig)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        plt.title(str_label)  
    plt.show()    


if __name__ == '__main__':
    x, y = create_data()
    model = fit_model (x, y)
    predict()
   
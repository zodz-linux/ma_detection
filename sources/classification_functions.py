#/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #evita los  warnings de keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np
db_path = (os.getcwd()+"/database/")

#################### DATASET UTILITIES  ###################################################################

def load_single_csv(csv_label):
    dataset = np.loadtxt(db_path+csv_label, delimiter=",")
    #print "Lectura de  datos cvs: ",csv_label[-29:], "  |  datos: ",len(dataset)
    return dataset

def load_list_csv(csv_list):
    vmas_set = np.zeros((1,11),dtype=np.float_)
    fmas_set = np.zeros((1,11),dtype=np.float_)
    for csv in csv_list:
        tmp = load_single_csv(csv)
        if "VMAS.csv" in csv:
            vmas_set = np.concatenate((vmas_set, tmp), axis=0)
        if "FMAS.csv" in csv:
            fmas_set = np.concatenate((fmas_set, tmp), axis=0)
    vmas_set = vmas_set[1:,:]
    fmas_set = fmas_set[1:,:]
    np.random.shuffle(vmas_set)
    np.random.shuffle(fmas_set)
    separator1 = int(len(vmas_set)*.8)
    separator2 = int(len(fmas_set)*.8)
    train_dataset = [vmas_set[:separator1,:], fmas_set[:separator2,:]]
    test_dataset  = [vmas_set[separator1:,:], fmas_set[separator2:,:]]
    print "\n\tConjuntos de datos cargados:\n\tVMAS: ",len(vmas_set),"\n\tFMAS: ",len(fmas_set)
    print "\tRelacion: ",len(vmas_set)/float(len(fmas_set))
    return train_dataset,test_dataset

#################### KERAS FUNCTIONS   #####################################################################
def generate_MLP(layers_tuple=[7,300,300,200,100]):
    # create model
    model = Sequential()
    model.add(Dense(layers_tuple[1], input_dim=layers_tuple[0],kernel_initializer="uniform", activation='relu'))
    for  layer in layers_tuple[2:]:
        model.add(Dense(layer, kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
    # Fit the model
    return model

#################### TREE FUNCTIONS   #####################################################################

def generate_tree():
    # Building Decision Tree - CART Algorithm (gini criteria)
    tree = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=5)
    return tree

def train_tree(tree,X,Y):
    tree.fit(X, Y)

#################### SVM FUNCTIONS   #####################################################################

def generate_svm():
    clf = svm.SVC()
    return clf







#

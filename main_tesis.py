#!/usr/bin/env python
# -*- coding: utf-8 -*-
##########################################
import os
import cv2
import numpy as np
from sources.clase_retina import *
from sources.clase_clasificadores import *
from sources.classification_functions import *
sources_path = os.getcwd()+"/sources/"
db_path      = os.getcwd()+"/database/"
##########################################


""" ------ EJECUCION ------"""
def Retina_Image_Process(images):
    for image in images:
        print("\n\n  Imagen cargada : "+Fore.BLUE +(image+"\n")),(Fore.RESET )
        retina = Retina()
        retina.Load_Image(image,"color")
        retina.Load_Ground_Truth()
        retina.Apply_Found_Cadidates()
        retina.Building_Work_Sets()
        #retina.Apply_Candidate_Label()
    print(Fore.YELLOW + str("\n\n\t\tImagenes Procesadas: "+str(len(images)))),(Fore.RESET )

def Classification_Process(train_dataset,test_dataset):
    model = clasificador()
    model.load_datasets(train_dataset,test_dataset)
    model.algorithm = "mlp"
    model.train()
    #model.test()
    #model.save_reports()
    #model.save_configuration() #here use  pickle
    return

def main():
    ### current variables
    images     = ["diaretdb1_image003.png"]
    images     = [img for img in os.listdir(db_path) if  ".png" in img ]
    csv_files  = [img for img in os.listdir(db_path) if  ".csv" in img ]
    images.sort()
    Retina_Image_Process(images)
    #train_dataset,test_dataset  = load_list_csv(csv_files) # list [vmas,fmas]
    #Classification_Process(train_dataset,test_dataset)




main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
##########################################
import os
import cv2
import numpy as np
from sources.functions import *
from sources.functions_v2 import *
from sources.clase_retina import *
from  matplotlib import  pyplot as plt
sources_path = (os.getcwd()+"/sources/")
##########################################
"""

#cargar imagen
msk = cv2.imread(sources_path+"mask.png",0)
img = cv2.imread(sources_path+"diaretdb1_image001.png")
cv2.imwrite("step00_original.png",img)

img = img[:,:,1]
cv2.imwrite("step01_green_layer.png",img)
img = cv2.medianBlur(img,5)
cv2.imwrite("step02_median_filtering.png",img)
#img[msk==0]=0



# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(10,10))
cl1 = clahe.apply(img)
cv2.imwrite('step03_clahe.png',cl1)

normalized =(cl1-float(cl1.min()))/(cl1.max()-cl1.min())
normalized*=255
normalized[normalized>200] =200
#normalized =(cl1-float(normalized.min()))/(normalized.max()-normalized.min())
#normalized*=200
cv2.imwrite('step04_normalized.png',normalized)

invert=(255-normalized.copy())



im = cv2.imread('step04_normalized.png', cv2.IMREAD_GRAYSCALE)

keypoints = Apply_Found_Blobs(im)

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("step05_found_blobs.png",im_with_keypoints)

puntos=[[int(point.size), int(point.pt[0]),int(point.pt[1])] for point in keypoints]
puntos.sort()

results = invert.copy()
results*=0
for p in puntos:
    p[0]+=5
    aux_size=int(p[0]/2.0)
    j = p[1] - aux_size
    i = p[2] - aux_size
    submatrix = (im[i:i+p[0],j:j+p[0]]).copy()

    results[i:i+p[0],j:j+p[0]] = submatrix

    print p[0], submatrix.shape
    #results[p[1]-w:p[1]+w,p[2]-w:p[2]+w] = subm.copy()
print len(puntos)


cv2.imwrite('step06_results.png',results)
"""

#Obtener_Candidatos(sources_path+image)

""" ------ EJECUCION ------"""
def Run_Retina_Process(paralellize=True,expert_number=2):
    images=["diaretdb1_image007.png","diaretdb1_image001.png","diaretdb1_image005.png","diaretdb1_image003.png","diaretdb1_image010.png","diaretdb1_image067.png"]
    images=["diaretdb1_image001.png"]

    print "\t  --> Preparando Imagenes <--"
    #Preprocessing_Images(images,paralellize)
    #contrast_path
    #dataset = []
    for image in images:
        retina = Retina()
        retina.Load_Image(sources_path+image,"color")
        retina.Apply_Found_Cadidates()


Run_Retina_Process()

#/usr/bin/python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
""" FUNCIONES INCLUIDAS

Apply_Found_Blobs          (img)
-> Configure_Blob_Detector ()
Subctract_Submatrix        (i,j,image,w_size)
Embed_Submatrix            (i,j,submatrix,image)
Single_Hu_Moments          (sub_matrix)
XML2Table                  (xml_label)
UsingMultiprocessing       (function,listArgs)

"""

def EuclidianDistance(x1,y1,x2,y2):
    dist = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return  dist


def Configure_Blob_Detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 16
    params.maxArea = 400

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = .6

    # Filter by Convexity
    params.filterByConvexity = False#True

    # Filter by Inertia
    params.filterByInertia = False#True

    return params


def Apply_Found_Blobs(img):
    params = Configure_Blob_Detector()

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
    return keypoints


def Subctract_Submatrix(f,c,image,w_size):
    """
    Se construye la ventana  sub_matrix con centro en (fila,columna)
    w_size corresponde al tamanho  que debe  tener la ventana al salir
    w_size es el diametro
    """
    aux_size=0
    if w_size%2 :
        aux_size= int((w_size-1)/2)
    else:
        aux_size= int(w_size/2)
    submatrix = (image[f-aux_size:(f+aux_size+1),c-aux_size:(c+aux_size+1)]).copy()
    return submatrix

def Embed_Submatrix(f,c,submatrix,image):
    img=image.copy()
    aux_size= ((submatrix.shape)[0]-1)/2
    img[f-aux_size:(f+aux_size+1),c-aux_size:(c+aux_size+1)]=submatrix.copy()
    return img


#---------------------  FEATURES EXTRACTION  ---------------------#

def external_normalization01(image):
    MaxValue        = (image).max()
    MinValue        = (image).min()
    NormalizedImage = (image-MinValue)/(float(MaxValue-MinValue))
    new_image = NormalizedImage.copy()
    return new_image

def get_Varianze(sub_matrix):
    data=(sub_matrix[sub_matrix>0]).tolist()
    min_data=float(min(data))
    max_data=float(max(data))
    data=np.array(data)
    normalized = (data-min_data)/(max_data-min_data)
    var = np.var(normalized)
    return var

def get_Median(sub_matrix):
    return np.median(sub_matrix)

def get_Contour(submatrix):
    sub_matrix = submatrix.copy()
    t=np.mean(sub_matrix)
    sub_matrix[sub_matrix<t]=255
    sub_matrix[sub_matrix<255]=0
    _,contours, hierarchy  = cv2.findContours(sub_matrix,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return  contours[0]
    return [0]

def get_Area(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    area = cv2.contourArea(contour)
    return area

def get_Perimeter(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    perimeter = cv2.arcLength(contour,True)
    return perimeter

def get_Aspect_Radio(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    return aspect_ratio

def get_Extend(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent

def get_Solidity(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0.0:
        solidity = float(area)/hull_area
        return solidity
    else:
        return 0.0

def get_Equivalent_Diameter(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    area = cv2.contourArea(contour)
    equi_diameter = np.sqrt(4*area/np.pi)
    return equi_diameter

def get_Orientation(sub_matrix):
    contour=get_Contour(sub_matrix)
    if len(contour) == 1:
        return 0
    (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
    return angle

def get_Extreme_Points(sub_matrix):
    cnt=get_Contour(sub_matrix)
    if len(cnt) == 1:
        return 0
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    return  leftmost,rightmost,topmost,bottommost

#---------------------- funciones anteriores -------------------#


def XML2Table(label):
    tree = ET.parse(label)#Leer  arbol xml
    observations=[]
    for node  in tree.iter("marking"): #Vamos  iterar sobre todos los  marcadores
        tmp_lesion   = [str(),int(),int(),int(),str(),int(),int(),list()]
        for  subchild01 in node.getchildren():
            # inicializacion de  atributos de la lista
            #obtenermos la informacion del centroide
            if subchild01.tag in "polygonregion":
                tmp_radius=0
                for subchild02 in subchild01:
                    if subchild02.tag in "centroid":
                        aux=(subchild02[0].text).split(",")
                        tmp_lesion[1] = int(aux[0])
                        tmp_lesion[2] = int(aux[1])
                    if subchild02.tag in "coords2d":
                        coordenates = [int(i) for i in ((subchild02.text).split(","))]
                        tmp_distance = abs(coordenates[0] - tmp_lesion[1])
                        if abs(coordenates[1] - tmp_lesion[2]) >  tmp_distance:
                            tmp_radius = max([tmp_radius,abs(coordenates[1] - tmp_lesion[2])])
                        tmp_radius = tmp_distance
                tmp_lesion[3] =tmp_radius
            elif subchild01.tag in "circleregion":
                for subchild02 in subchild01:
                    if subchild02.tag in "centroid":
                        aux=(subchild02[0].text).split(",")
                        tmp_lesion[1] = int(aux[0])
                        tmp_lesion[2] = int(aux[1])
                    if subchild02.tag in "radius":
                        tmp_lesion[3] = int(subchild02.text)
            elif subchild01.tag in "representativepoint":
                aux = (subchild01[0].text).split(",")
                tmp_lesion[5] = int(aux[0])
                tmp_lesion[6] = int(aux[1])
                if tmp_lesion[1] == 0:
                    tmp_lesion[1] = int(aux[0])
                if tmp_lesion[2] == 0:
                    tmp_lesion[2] = int(aux[1])
                #print subchild01[0].text
            elif subchild01.tag in "confidencelevel":
                tmp_lesion[4]=(subchild01.text)
                pass
            elif  subchild01.tag in  "markingtype":
                if subchild01.text in "Haemorrhages":
                    tmp_lesion[0] = "haemorrhage"
                elif subchild01.text in "Red_small_dots":
                    tmp_lesion[0] = "small_dot"
        if tmp_lesion[0] != "":
            tmp_lesion.pop()
            observations.append(tmp_lesion)
    final_observations=[instance for instance in observations if instance[0] == "small_dot"]

    return final_observations

def BorderExtensorMatrix(Image,BorderExtension=1):
    # entrada: objeto matriz imagen
    # 1.- Obtener las dimensiones de la matriz
    # 2.- Inicializar matriz en ceros de dimension  h+2,w+2
    # 3.- Incruztar matriz original en la nueva matriz
    H,W=Image.shape #1
    Matriz=np.zeros(((H+(BorderExtension*2)),(W+(BorderExtension*2))),dtype=int)#2
    Matriz[BorderExtension:H+BorderExtension,BorderExtension:W+BorderExtension]=Image.copy()#3
    return  Matriz

def UsingMultiprocessing(function,listArgs):
    """ Using the multiprocessing module  python share data and function"""
    Ncpus=multiprocessing.cpu_count()            #assigna el numero de  procesadores disponibles
    if Ncpus >=2:
        Ncpus = Ncpus-1
    pool = multiprocessing.Pool(processes=Ncpus) #crea un objeto multiprocesador con Ncpus
    pool_outputs = pool.map(function,listArgs )  #mapea la function en la lista de argumentos
    pool.close() #cierra (bloquea) el objeto pool
    pool.join()  #mantiene el bloqueo hasta la terminacion de los procesos
    return pool_outputs

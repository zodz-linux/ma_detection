# -*- coding: utf-8 -*-
#!/usr/bin/env python
##########################################
import os
import cv2
import numpy as np
from retina_functions import *
#from functions_v2 import *
np.seterr(invalid='ignore')
sources_path = (os.getcwd()+"/sources/")
db_path = (os.getcwd()+"/database/")
details_path = (os.getcwd()+"/imagenes_detalladas/")
from colorama import Fore, init
init()
##########################################

class Retina(object):
    def __init__(self):
        tmp   =  (cv2.imread(db_path+"diaretdb1_image001.png",1))
        tmp   = tmp[:,:,1]
        self.size                = tmp.shape
        self.border_extension    = 4
        self.image_label         = str()
        self.image               = tmp.copy() # guarda el ultimo  metodo realizado
        self.clahe               = tmp.copy() # guarda  una imagen binarizada mediante un  umbral
        self.hue                 = tmp.copy()
        self.i_green_plane       = tmp.copy()
        self.i_green_normalized  = tmp.copy()
        self.i_green_median      = tmp.copy()
        self.i_hue               = tmp.copy()
        self.i_red_plane         = tmp.copy()
        self.i_red_normalized    = tmp.copy()
        self.Candidates          = list()
        self.GroundTruth1        = list()
        self.GroundTruth2        = list()
        self.GroundTruth3        = list()
        self.GroundTruth4        = list()
        print "\t*Instancia Retina construida"

    def Load_Image(self,image_label,color):
        self.image_label=image_label[-22:-4]
        tmp_label=details_path+self.image_label #Cargar imagen
        tmp   = np.zeros((1152, 1500),dtype=np.int)
        if  color == "grey":
             tmp  = (cv2.imread(image_label,0))
             print "\t*Imagen en escala de grises cargada"

        if  color == "color":
            tmp   = (cv2.imread(db_path+image_label,1))
            brightHSV = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            self.i_hue,_,_ = cv2.split(brightHSV)
            self.i_red_plane=tmp[:,:,2]
            tmp   = tmp[:,:,1]
            self.i_red_normalized = external_normalization01(self.i_red_plane)

            cv2.imwrite(tmp_label+"_10_Capa_Roja.png",self.i_red_plane)
            cv2.imwrite(tmp_label+"_11_Capa_Roja_Normalizada.png",(self.i_red_normalized)*255)
            cv2.imwrite(tmp_label+"_12_HUE.png",self.i_hue)

            print "\t*Imagen a color recibida"
        if (self.image).max() == 0:
            self.__init__()
            print "Error al cargar la imagen. Imagen por defecto cargada."
        self.image = tmp.copy() # guarda el ultimo  metodo realizado

    def Load_Ground_Truth(self):
        """Esta funcion  recibe un fichero con extension xml y
        regresa  un dataframe pandas con las caracteristicas importantes"""
        for num_expert in [1,2,3,4]:
            xml_file = db_path+self.image_label+"_0"+str(num_expert)+".xml"
            tabla = XML2Table(xml_file)
            for r in xrange(len(tabla)):
                for i in [1,2,5,6]:
                    (tabla[r])[i] += self.border_extension
            if num_expert == 1:
                self.GroundTruth1 = tabla
            elif num_expert == 2:
                self.GroundTruth2 = tabla
            elif num_expert == 3:
                self.GroundTruth3 = tabla
            elif num_expert == 4:
                self.GroundTruth4 = tabla
        print "\t*Metodo: Lectura de GroundTruth"

    def Reload_Temporal(self):
        """ Este metodo recarga la  imagen como objeto de
        cv2. Escribiendo-Leyendo. Este metodo es un parche"""
        cv2.imwrite(sources_path+".tmp.png",self.image.copy())
        self.image= cv2.imread(sources_path+'.tmp.png',0)

    def Apply_Normalize(self):
        #----------------------------------------------------------#
        """ Normaliza los datos (rango 0,1)
        y la guarda en el atributo normalized """
        print "\t*Metodo: Normalizacion"
        #----------------------------------------------------------#
        MaxValue        = (self.image).max()
        MinValue        = (self.image).min()
        NormalizedImage = (self.image-MinValue)/(float(MaxValue-MinValue))
        self.image = NormalizedImage.copy()*255

    def Apply_Mean_Filter(self,kernel_size=5):
        self.Reload_Temporal()
        print "\t*Metodo: Filtrado de Media"
        self.image = cv2.medianBlur(self.image.copy(),kernel_size)

    def Apply_CLAHE(self):
        print "\t*Metodo: Histograma Adaptativo  CLAHE"
        self.Reload_Temporal()
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(10,10))
        cl1 = clahe.apply(self.image)
        self.image = cl1.copy()
        self.clahe = cl1.copy()

    def Apply_Remove_Threshold(self):
        print "\t*Metodo: Remover  valores superiores"
        ################
        self.image[self.image>200] =200

    def Apply_Ring_Validation(self,f,c,w_size):
        W_out = Subctract_Submatrix(f,c,self.image,w_size+self.border_extension)
        W_out[self.border_extension:f+w_size,self.border_extension:c+w_size]=0
        #Validacion de dimension de  matriz
        rows,columns=W_out.shape
        if rows!=columns: return False
        var=get_Varianze(W_out)
        if (var < 0.07):
            return True
        else:
            return False

    def Apply_Found_Cadidates(self):
        print(Fore.YELLOW +"\tInicia busqueda de candidatos"),(Fore.RESET )
        tmp_label=details_path+self.image_label #Cargar imagen

        self.i_green_plane = self.image.copy()
        cv2.imwrite(tmp_label+"_00_Original_Green.png",self.image)

        self.Apply_Normalize() # paso valido
        self.i_green_normalized = external_normalization01(self.image)
        cv2.imwrite(tmp_label+"_01_Normalized.png",self.image)

        self.Apply_Mean_Filter() # paso valido
        self.i_green_median = self.image.copy()
        cv2.imwrite(tmp_label+"_02_Mean_Filter.png",self.image)

        self.Apply_CLAHE() # paso valido
        self.clahe = self.image.copy()
        cv2.imwrite(tmp_label+"_03_CLAHE.png",self.image)


        self.Apply_Normalize() # paso valido
        self.Apply_Remove_Threshold() # paso valido
        cv2.imwrite(tmp_label+"_04_Superiores_Igualados.png",self.image)

        self.image=BorderExtensorMatrix(self.image,self.border_extension)
        cv2.imwrite(tmp_label+"_05_Borde_Amentado.png",self.image)

        self.Reload_Temporal() # paso valido
        keypoints = Apply_Found_Blobs(self.image) # paso valido
        im_with_keypoints = cv2.drawKeypoints(self.image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(tmp_label+"_06_Blobs_Encontrados.png",im_with_keypoints)


        puntos=[[int(point.size), int(point.pt[0]),int(point.pt[1])] for point in  keypoints]

        """parte ilustrativa del recorte de puntos"""
        #Subctract_Submatrix(i,j,image,w_size)
        #Embed_Submatrix(i,j,submatrix,image)
        results = self.image.copy()
        results*=0
        for p in puntos:
            w_size = p[0]#+2 #Agregamos un pixel de grosor adicional
            i,j = p[2],p[1]
            submatrix=Subctract_Submatrix(i,j,self.image,w_size)
            results = Embed_Submatrix(i,j,submatrix,results)
        cv2.imwrite(tmp_label+"_07_Blobs Aislados.png",results)
        """parte ilustrativa del recorte de puntos"""


        puntos=[]
        for point in keypoints:
            aux= [int(point.size), int(point.pt[0]),int(point.pt[1]),0]
            if (self.Apply_Ring_Validation(aux[1],aux[2],aux[0])):
                puntos.append(aux)
        self.Candidates = tuple(puntos) # paso valido
        print "\t*Reduccion de candidatos por Doble anillo: "\
        +(Fore.YELLOW+(str(len(keypoints))+" --> "+str(len(puntos)))),(Fore.RESET )


        """parte ilustrativa del recorte de puntos"""
        results = self.image.copy()
        results*=0
        for p in puntos:
            w_size = p[0]#+2 #Agregamos un pixel de grosor adicional
            i,j = p[2],p[1]
            submatrix=Subctract_Submatrix(i,j,self.image,w_size)
            results = Embed_Submatrix(i,j,submatrix,results)
        cv2.imwrite(tmp_label+"_08_Filtrado_anillo.png",results)
        """parte ilustrativa del recorte de puntos"""

        """parte ilustrativa del recorte de puntos"""
        results = self.image.copy()
        results*=0
        t=70
        for p in puntos:
            w_size = p[0]#+2 #Agregamos un pixel de grosor adicional
            i,j = p[2],p[1]
            submatrix=Subctract_Submatrix(i,j,self.image,w_size)
            ###################################################
            t=np.median(submatrix)
            submatrix[submatrix<t]=255
            submatrix[submatrix<255]=0
            ###################################################
            results = Embed_Submatrix(i,j,submatrix,results)
        cv2.imwrite(tmp_label+"_09_contornos.png",results)
        """parte ilustrativa del recorte de puntos"""

    def Apply_Candidate_Label(self):
        experts=(self.GroundTruth1+self.GroundTruth2+self.GroundTruth3+self.GroundTruth4)
        labeled = []
        for  candidate in self.Candidates:
            tmp_candidate = candidate[:]
            for expert in experts:
                if EuclidianDistance(candidate[1],candidate[2],expert[1],expert[2]) < 5:
                    #print "\n\nCoincidencia encontrada:\n\tExperto:   ",expert[1:4]
                    #print "\tCandidato: ",candidate[1],candidate[2],candidate[0]
                    labeled.append(candidate)

        print "\n\nCandidatos Registrados:", len(self.Candidates)
        print "Candidatos que Coinciden:", len(labeled)
        print "\nObservaciones Experto 1:",len(self.GroundTruth1)
        print "Observaciones Experto 2:",len(self.GroundTruth2)
        print "Observaciones Experto 3:",len(self.GroundTruth3)
        print "Observaciones Experto 4:",len(self.GroundTruth4)


    def Feature_Extraction(self,i,j,w_size):
        feature_list = list()
        sub_matrix   = Subctract_Submatrix(i,j,self.image,(w_size))
        # ----------------------------------------------------- #
        area         = get_Area(sub_matrix)
        feature_list.append(area)
        perimeter    = get_Perimeter(sub_matrix)
        feature_list.append(perimeter)
        aspect_ratio = get_Aspect_Radio(sub_matrix)
        feature_list.append(aspect_ratio)
        extend       = get_Extend(sub_matrix)
        feature_list.append(extend)
        solidity     = get_Solidity(sub_matrix)
        feature_list.append(solidity)
        equi_diameter= get_Equivalent_Diameter(sub_matrix)
        feature_list.append(equi_diameter)
        #orientation  = get_Orientation(sub_matrix)
        #feature_list.append(orientation)
        #l,r,t,b      = get_Extreme_Points(sub_matrix) #l,r,t,b = leftmost,rightmost,topmost,bottommost
        #feature_list+=[l,r,t,b]
        var          = get_Varianze(sub_matrix)
        feature_list.append(var)
        median       = get_Median(sub_matrix)
        feature_list.append(median)
        # ----------------------------------------------------- #
        #hu_moments   = list(cv2.HuMoments(cv2.moments(sub_matrix)).flatten())

        return feature_list


    def Building_Work_Sets(self,FP=250):
        print"\t*Metodo: Construccion de conjuntos",(Fore.YELLOW + " VMAS y FMAS "),(Fore.RESET )
        #print "\t*Construccion  de conjuntos"
        experts=(self.GroundTruth1,self.GroundTruth2,self.GroundTruth3,self.GroundTruth4)
        expert_instances    = []
        generated_instances = []
        for e in xrange(4):
            for instance in experts[e]:
                f,c,radio= instance[1],instance[2],(instance[3])*2,
                tmp_list = [f,c,radio]
                #working here!
                sub_matrix   = Subctract_Submatrix(f,c,self.image,radio)
                if np.mean(sub_matrix) > 10:
                    current_features= self.Feature_Extraction(f,c,radio)
                    current_features.append(1)
                    expert_instances.append(tmp_list+current_features)
            ##### Generador de  Observaciones
            Rows,Columns = (self.image).shape
            generator=lambda low,upper:np.random.randint(low,upper)
            while len(generated_instances) < FP:
                radio = generator(7,21)
                f     = generator(radio,Rows-radio+1)
                c     = generator(radio,Columns-radio+1)
                subm  = Subctract_Submatrix(f,c,self.image,radio)
                for expert in experts:
                    dist  = EuclidianDistance(f,c,instance[1],instance[2])
                if np.mean(subm) > 40 and (dist > radio):
                    tmp_list=[f,c,radio]
                    sub_matrix   = Subctract_Submatrix(f,c,self.image,radio)
                    if np.min(sub_matrix) > 0:
                        current_features= self.Feature_Extraction(f,c,radio)
                        current_features.append(0)
                        generated_instances.append(tmp_list+current_features)
        ##### Division de conjuntos
        np.random.shuffle(expert_instances)
        np.random.shuffle(generated_instances)
        ################ GRABAR EN  ARCHIVOS ##################

        f1=open(db_path+self.image_label+"_VMAS.csv","wb")
        for instance in expert_instances:
            f1.write(str(instance)[1:-1])
            f1.write("\n")
        f1.close()

        f2=open(db_path+self.image_label+"_FMAS.csv","wb")
        for instance in generated_instances:
            f2.write(str(instance)[1:-1])
            f2.write("\n")
        f2.close()



#

#!/usr/bin/env python
# -*- coding: utf-8 -*-
##########################################
from classification_functions import *
from colorama import Fore, init
init()

class clasificador(object):
    def __init__(self):
        self.algorithm    = "mlp"
        self.args_list    = list()
        self.model        = self.generar_modelo()
        self.X_train      = np.array([])
        self.Y_train      = np.array([])
        self.X_test       = np.array([])
        self.Y_test       = np.array([])
        self.kfold_dataset= list()
        self.num_features = 7

    def load_datasets(self,train,test):
        #unpack  dataset [0] vmas [1] fmas
        vmas_train,fmas_train = train[0],train[1]
        #vmas_test,fmas_test = test[0],test[1]
        muestras=list()
        for it in  xrange(10): #numero de conjuntos  para entrenar
            np.random.shuffle(vmas_train)
            np.random.shuffle(fmas_train)
            #tmp=[vmas_train[:20],fmas_train[:20]]
            tmp=np.concatenate((vmas_train[:100,:],fmas_train[:100,:]), axis=0)
            np.random.shuffle(tmp)
            muestras.append(tmp)
        self.kfold_dataset=muestras
        test_dataset = np.concatenate((test[0],test[1]), axis=0)
        np.random.shuffle(test_dataset)
        self.X_test = test_dataset[:,3:-1]
        self.Y_test = test_dataset[:,-1]

    def generar_modelo(self):
        if  self.algorithm == "mlp" :
            self.model= generate_MLP()
            print"\tModelo Generado: ",(Fore.YELLOW + str("Perceptron Multicapa")),(Fore.RESET)
        if self.algorithm == "tree":
            self.model=generate_tree()
            print"\tModelo Generado: ",(Fore.YELLOW + str("Arbol")),(Fore.RESET)
        if self.algorithm == "svm":
            self.model=generate_svm()
            print"\tModelo Generado: ",(Fore.YELLOW + str("Maquina de Vector Soporte")),(Fore.RESET)

    def train(self):
        print "\tMetodo de entrenamiendo del MLP"
        self.generar_modelo()
        if self.algorithm == "mlp":
            #aqui corre el algoritmo de  cross validation
            contador=0
            for muestra in self.kfold_dataset:
                self.X_train = muestra[:,3:-1]
                self.Y_train = muestra[:,-1]
                self.model.fit(self.X_train, self.Y_train, epochs=20, batch_size=10)
        scores = self.model.evaluate(self.X_test, self.Y_test)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        # calculate predictions
        predictions = self.model.predict(self.X_test)
        print self.X_test[0],self.Y_test[0]
        # round predictions
        print predictions
        rounded = [int(round(x[0])) for x in predictions]

        detectados=0
        no_detectados=0
        contador=0
        no_lesiones=0
        for i in xrange(len(rounded)):
            if int(self.Y_test[i]) == 1 :
                if rounded[i] == 1:
                    detectados+=1
                else:
                    no_detectados+=1
                contador+=1

        print "\tNumero de muestras:",len(self.Y_test)
        print "\tLesiones: ", contador
        print "\tNo lesiones: ",no_lesiones
        print "\tdetectadas: ",detectados
        print "\tNo detectadas: ",no_detectados
        print "\tRendimiento Real: ",(detectados/float(contador))*100,"%"


    #def test(self):

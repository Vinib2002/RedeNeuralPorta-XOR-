# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:22:12 2022

@author: vini
"""
#Importação das Blibliotecas juntamente com o tipo de função
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

rede = buildNetwork(2, 3, 1)#Criação da Rede, rede com dois neurônios de entrada,3 ocultos e 1 de saida
base = SupervisedDataSet(2, 1)#base de dados com 2 entradas e 1 saida
#Definindo valores de entradas e saidas
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))
#Treinamento e definição de paramentros não obrigatorios
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01,
                              momentum = 0.06)
#Exibir o erro a cada 2000 epocas
for i in range(1, 30000):
    erro = treinamento.train()
    if i % 2000 == 0:
        print("Erro: %s" % erro)
#Dificilmente geram valores exatos, mas percebam a aproximação deles dos valores esperados      
print('O valor previsto será de: ',rede.activate([0, 0]))
print('O valor previsto será de: ',rede.activate([1, 0]))
print('O valor previsto será de: ',rede.activate([0, 1]))
print('O valor previsto será de: ',rede.activate([1, 1]))


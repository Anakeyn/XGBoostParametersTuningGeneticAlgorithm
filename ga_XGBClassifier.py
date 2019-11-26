#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 22:33:49 2019

@author: Pierre
"""

# Code from Chapter 10  p211 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.
# Stephen Marsland, 2008, 2014 
# The Genetic algorithm
# Comment and uncomment fitness functions as appropriate (as an import and the fitnessFunction variable)
# voir original sur Github:
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch10/ga.py
# Modifications par Pierre Rouarch :
# de ga Standard à un ga pour trouver les paramètres de réglages de XGBClassifier
# Inspiré par Mohit Jain - "Hyperparameter tuning in XGBoost using genetic algorithm"
# https://github.com/mjain72/Hyperparameter-tuning-in-XGBoost-using-genetic-algorithm
#on a aussi enlevé le plot qui peut être fait à l'extérieur de la classe.

import numpy as np
import random

class ga_XGBClassifier:

	def __init__(self,fitnessFunction,nEpochs, X, y, X_train, X_test, y_train, y_test, populationSize=100,crossover='un',nElite=4,tournamentOK=True):
		""" Constructor"""
		
		self.stringLength = 7 #ici  on a 7 paramètres  donc 7 gènes
		
		# Population size should be even
		if np.mod(populationSize,2)==0:
			self.populationSize = populationSize
		else:
			self.populationSize = populationSize+1
        
		#MutationProb non utilisé en fait la mutation est de 1/7 car on mute un seul paramètres à la fois
	 	  
		self.nEpochs = nEpochs     #nombre de générations 
		self.bestfit = np.zeros(self.nEpochs) #raz de bestfit 
		self.bestParams = np.zeros((self.nEpochs,7)) #raz de bestParams
		self.fitness = np.zeros(self.populationSize)  #pour conserver les derniers fitness #pas sur que cela soit utile


		self.fitnessFunction = fitnessFunction #ici calcul du Test Score de XGBoostClassifier

		self.crossover = crossover  #type de crossover 'un' uniforme ou 'sp'
		self.nElite = nElite              #nombre de parents conservés dans l'élitisme
		self.tournamentOK = tournamentOK  #Est-ce que l'on fait le tournoi ?
        
        ####### Données pour XGBClassifier
		self.X = X
		self.y = y
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
        
        
        #### Construction de la population #######
		self.learningRate = np.empty([self.populationSize, 1])
		self.nEstimators = np.empty([self.populationSize, 1], dtype = np.uint8)
		self.maxDepth = np.empty([self.populationSize, 1], dtype = np.uint8)
		self.minChildWeight = np.empty([self.populationSize, 1])
		self.gammaValue = np.empty([self.populationSize, 1])
		self.subSample = np.empty([self.populationSize, 1])
		self.colSampleByTree =  np.empty([self.populationSize, 1])

        #valeurs de départ au hasard.
		for i in range(self.populationSize):
			 #print(i)
			 self.learningRate[i] = round(random.uniform(0.01, 1), 2)
			 self.nEstimators[i] = int(random.randrange(10, 2000, step = 25))
			 self.maxDepth[i] = int(random.randrange(1, 15, step= 1))
			 self.minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
			 self.gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
			 self.subSample[i] = round(random.uniform(0.01, 1.0), 2)
			 self.colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)
    
		self.population = np.concatenate((self.learningRate, self.nEstimators, self.maxDepth, self.minChildWeight, self.gammaValue, self.subSample, self.colSampleByTree), axis= 1)

    #boucle	
	def runGA(self):
		"""The basic loop"""
        #commenter les print si vous trouvez qu'il y a trop d'affichage
		print("ca tourne")

		self.bestfit = np.zeros(self.nEpochs) #raz de bestfit 
		self.bestParams = np.zeros((self.nEpochs,7)) #raz de bestParams

		for i in range(self.nEpochs):
			print("boucle génération = "+str(i))
            # Compute fitness of the population
			fitness = (self.fitnessFunction)(self.population, self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test )
			print ("Liste des fitness pour cette génération ", i, fitness)
            
            
			if (np.mod(i,1)==0):
				print("Meilleurs paramètres de cette génération ", i, self.population[np.argmax(fitness),:])
				print("Meilleur Fitness de cette génération ", i, max(fitness))	#il affiche le meilleur toutes les x iterations
			
            #sauvegarde dans l'objet pour utilisation extérieure.(notamment plotting)
			self.bestParams[i] = self.population[np.argmax(fitness),:]  #meilleurs paramètres de cette génération
			self.bestfit[i] = max(fitness) #on conserve le meilleur fitness de la génération i pour faire un graphique
			self.fitness = fitness #on conserve les derniers fitness de cette génération   #pas vraiment utile
			        
            #Pick parents -- can do in order since they are randomised
            #récupere les parents les plus aptes 
			newPopulation = self.fps(self.population,fitness)  #prend les individus les plus aptes en priorité.

			# Apply the genetic operators
			if self.crossover == 'sp':  #croisement à partir d'un point
				newPopulation = self.spCrossover(newPopulation)
			elif self.crossover == 'un':  #croisement uniforme
				newPopulation = self.uniformCrossover(newPopulation)
            
            #Mutation
			newPopulation = self.mutation(newPopulation)  #mutation sur un des paramètres au hasard
            
            
			# Apply elitism 
			if self.nElite>0:
				newPopulation = self.elitism(self.population,newPopulation,fitness)
            #apply tournoi   
			if self.tournamentOK :
				newPopulation = self.tournament(self.population,newPopulation,fitness,self.fitnessFunction, self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test)
	
			self.population = newPopulation  #nouvelle population pour le tour suivant

        
	#sélection des parents
	def fps(self,population,fitness):   #echantillonage pour reproduction.

		print("Recupération des parents")
		print("fitness en entrée", fitness)
		# Scale fitness by total fitness
		fitness = fitness/np.sum(fitness)
		fitness = 10*fitness/fitness.max()
		print("fitness transformé", fitness)
		# Put repeated copies of each string in according to fitness
		# Deal with strings with very low fitness
		j=0
		while np.round(fitness[j])<1:
			j = j+1
		
		print("Produit de Kronecker ", j, "fitness ", fitness[j])        
        
        #Produit de Kronecker : echantillonage au hasard à probabilité inégale (parce que aptitude)
        #simule la sélection naturelle. Plus le chromosome est apte plus il a de chance  d'être sélectionné.
		newPopulation = np.kron(np.ones((int(np.round(fitness[j])),1)),population[j,:]) 

		# Add multiple copies of strings into the newPopulation  #ici les suivants.
		for i in range(j+1,self.populationSize):
			print("Add multiple copies i = ", i)
			if np.round(fitness[i])>=1:
				print("Avec fitness : ", fitness[i])
				newPopulation = np.concatenate((newPopulation,np.kron(np.ones((int(np.round(fitness[i])),1)),population[i,:])),axis=0)

		# Shuffle the order (note that there are still too many) on prend les n premiers : populationSize
		indices = list(range(np.shape(newPopulation)[0]))
		np.random.shuffle(indices)
		newPopulation = newPopulation[indices[:self.populationSize],:]
		return newPopulation	

    #croisement à partir d'un point
	def spCrossover(self,population):
        
		print("Single point crossover")
		# Single point crossover
		newPopulation = np.zeros(np.shape(population))
		crossoverPoint = np.random.randint(0,self.stringLength,self.populationSize)
		for i in range(0,self.populationSize,2):
			newPopulation[i,:crossoverPoint[i]] = population[i,:crossoverPoint[i]]
			newPopulation[i+1,:crossoverPoint[i]] = population[i+1,:crossoverPoint[i]]
			newPopulation[i,crossoverPoint[i]:] = population[i+1,crossoverPoint[i]:]
			newPopulation[i+1,crossoverPoint[i]:] = population[i,crossoverPoint[i]:]
		return newPopulation
    
    #croisement uniforme
	def uniformCrossover(self,population):
        
		print("Uniform crossover")
		# Uniform crossover
		newPopulation = np.zeros(np.shape(population))
		which = np.random.rand(self.populationSize,self.stringLength)
		which1 = which>=0.5
		for i in range(0,self.populationSize,2):
			newPopulation[i,:] = population[i,:]*which1[i,:] + population[i+1,:]*(1-which1[i,:])
			newPopulation[i+1,:] = population[i,:]*(1-which1[i,:]) + population[i+1,:]*which1[i,:]
		return newPopulation
    
	#mutation	
	def mutation(self,population):
		#Attention ici la mutation se fait sur un seul paramètre/gène  à la fois.
		#Define minimum and maximum values allowed for each parameter

		minMaxValue = np.zeros((7, 2))  #pour enregistrer les valeurs. Min et Max
    
		minMaxValue[0:] = [0.01, 1.0] #min/max learning rate
		minMaxValue[1, :] = [10, 2000] #min/max n_estimators
		minMaxValue[2, :] = [1, 15] #min/max depth
		minMaxValue[3, :] = [0, 10] #min/max child_weight
		minMaxValue[4, :] = [0.01, 10.0] #min/max gamma
		minMaxValue[5, :] = [0.01, 1.0] #min/maxsubsample
		minMaxValue[6, :] = [0.01, 1.0] #min/maxcolsample_bytree
 
		# Mutation changes a single gene in each offspring randomly.
		mutationValue = 0
		parameterSelect = np.random.randint(0, 7, 1)  #parametre au hasard
		print("Mutation Paramètre sélectionné : "+str(parameterSelect))  
		if parameterSelect == 0: #learning_rate
			mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
		if parameterSelect == 1: #n_estimators
			mutationValue = np.random.randint(-200, 200, 1)
		if parameterSelect == 2: #max_depth
			mutationValue = np.random.randint(-5, 5, 1)
		if parameterSelect == 3: #min_child_weight
			mutationValue = round(np.random.uniform(-5, 5), 2)  
		if parameterSelect == 4: #gamma
			mutationValue = round(np.random.uniform(-2, 2), 2)
		if parameterSelect == 5: #subsample
			mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
		if parameterSelect == 6: #colsample
			mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
  
		#introduce mutation by changing one parameter, and set to max or min if it goes out of range
		for idx in range(population.shape[0]):
			population[idx, parameterSelect] = population[idx, parameterSelect] + mutationValue
			if(population[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
			   population[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
			if(population[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
			   population[idx, parameterSelect] = minMaxValue[parameterSelect, 0]    
		return population

 
    
    #on choisit les nElite meilleurs - 4 par défaut
	def elitism(self,oldPopulation,population,fitness):
		print("Elitisme")
		best = np.argsort(fitness)
		best = np.squeeze(oldPopulation[best[-self.nElite:],:])
		indices = list(range(np.shape(population)[0]))
		np.random.shuffle(indices)
		population = population[indices,:]
		population[0:self.nElite,:] = best
		return population
    
    #les 2 meilleurs parmi les parents et leurs enfants.
	def tournament(self,oldPopulation,population,fitness,fitnessFunction, X, y, X_train, X_test, y_train, y_test):
		print("Début de Tournoi")
		newFitness = (self.fitnessFunction)(population, X, y, X_train, X_test, y_train, y_test)
		print("Nouveau Fitness", newFitness)
		print("on va tourner ", np.shape(population)[0], " fois par pas de 2" )
		for i in range(0,np.shape(population)[0],2):
			f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=0)
			indices = np.argsort(f)
			if indices[-1]<2 and indices[-2]<2:
				population[i,:] = oldPopulation[i,:]
				population[i+1,:] = oldPopulation[i+1,:]
			elif indices[-1]<2:
				if indices[0]>=2:
					population[i+indices[0]-2,:] = oldPopulation[i+indices[-1]]
				else:
					population[i+indices[1]-2,:] = oldPopulation[i+indices[-1]]
			elif indices[-2]<2:
				if indices[0]>=2:
					population[i+indices[0]-2,:] = oldPopulation[i+indices[-2]]
				else:
					population[i+indices[1]-2,:] = oldPopulation[i+indices[-2]]
		print("Fin de Tournoi")
		return population
			

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 18:39:18 2019

@author: Pierre
"""
##########################################################################
# XGBoostParametersTuningGeneticAlgorithm - modifié le 26/11/2011
# Auteur : Pierre Rouarch - Licence GPL 3
# Exemple d'utilisation d'un algorithme génétique pour améliorer les performances 
# D'un algorithme XGBoost
# la fonction de fitness ou d'évaluation  se base sur le test score
#####################################################################################

###################################################################
# On démarre ici 
###################################################################
#Chargement des bibliothèques générales utiles
import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
import pandas as pd  #pour les Dataframes ou tableaux de données
import seaborn as sns #graphiques étendues
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#pour les scores
#from sklearn.metrics import f1_score
#from sklearn.metrics import matthews_corrcoef

print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/MyPath"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif



#############################################################
# On revient au Machine Learning
#####################################################################


#Lecture des données suite  à scraping ############
dfQPPS8 = pd.read_csv("dfQPPS7.csv")
dfQPPS8.info(verbose=True) # 12194 enregistrements.    
dfQPPS8.reset_index(inplace=True, drop=True) 

#Variables explicatives
X =  dfQPPS8[['isHttps', 'level', 
             'lenWebSite', 'lenTokensWebSite',  'lenTokensQueryInWebSiteFrequency',  'sumTFIDFWebSiteFrequency',            
             'lenPath', 'lenTokensPath',  'lenTokensQueryInPathFrequency' , 'sumTFIDFPathFrequency',  
              'lenTitle', 'lenTokensTitle', 'lenTokensQueryInTitleFrequency', 'sumTFIDFTitleFrequency',
              'lenDescription', 'lenTokensDescription', 'lenTokensQueryInDescriptionFrequency', 'sumTFIDFDescriptionFrequency',
              'lenH1', 'lenTokensH1', 'lenTokensQueryInH1Frequency' ,  'sumTFIDFH1Frequency',        
              'lenH2', 'lenTokensH2',  'lenTokensQueryInH2Frequency' ,  'sumTFIDFH2Frequency',          
              'lenH3', 'lenTokensH3', 'lenTokensQueryInH3Frequency' , 'sumTFIDFH3Frequency',
              'lenH4',  'lenTokensH4','lenTokensQueryInH4Frequency', 'sumTFIDFH4Frequency', 
              'lenH5', 'lenTokensH5', 'lenTokensQueryInH5Frequency', 'sumTFIDFH5Frequency', 
              'lenH6', 'lenTokensH6', 'lenTokensQueryInH6Frequency', 'sumTFIDFH6Frequency', 
              'lenB', 'lenTokensB', 'lenTokensQueryInBFrequency', 'sumTFIDFBFrequency', 
              'lenEM', 'lenTokensEM', 'lenTokensQueryInEMFrequency', 'sumTFIDFEMFrequency', 
              'lenStrong', 'lenTokensStrong', 'lenTokensQueryInStrongFrequency', 'sumTFIDFStrongFrequency', 
              'lenBody', 'lenTokensBody', 'lenTokensQueryInBodyFrequency', 'sumTFIDFBodyFrequency', 
              'elapsedTime', 'nbrInternalLinks', 'nbrExternalLinks' ]]  #variables explicatives

X.info()
y =  dfQPPS8['group']

#on va scaler
scaler = StandardScaler()
scaler.fit(X)


X_Scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
X_Scaled.info()

#on choisit random_state = 42 en hommage à La grande question sur la vie, l'univers et le reste
#dans "Le Guide du voyageur galactique"   par  Douglas Adams. Ceci afin d'avoir le même split
#tout au long de notre étude.
X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y, random_state=42)




#########################################################################
# XGBOOST  
##########################################################################
#xgboost avec parametres standards par défaut

myXGBoost =   XGBClassifier().fit(X_train,y_train)
print("Training set score: {:.3f}".format(myXGBoost.score(X_train,y_train))) 
print("Test set score: {:.3f}".format(myXGBoost.score(X_test,y_test))) 
baseTestScore = myXGBoost.score(X_test,y_test)

#parametres par défaut    
myXGBoost.get_xgb_params()


###########################################################################
# réglage fin des paramètres de XGBClassifier
# On va utiliser un algorithme génétique pour rechercher des meilleurs 
# paramètres de XGBCLassifier
# Modification du programme ga de Stephen Marsland  
# inspiré aussi de mohit jain  
# passage de xgb.train en XGBClassifier
# Remarque : 'n_estimators'  n'est pas pris en compte dans xgb.train()
###########################################################################


###########################  Fonction de Fitness utilisée ici 
#La fonction d'évaluation est basée sur le  test score  
#on calcule le F1 score pour chaque XGBClassifier
def train_populationClassifier(population, X, y, X_train, X_test, y_train, y_test):
    print("Fitness Function")
    TestScore = []
    for i in range(population.shape[0]):
        print("Fitness Boucle dans la population "+str(i))
        param = { 'objective':'binary:logistic',
              'learning_rate': population[i][0],
              'n_estimators': int(population[i][1]), 
              'max_depth': int(population[i][2]), 
              'min_child_weight': population[i][3],
              'gamma': population[i][4], 
              'subsample': population[i][5],
              'colsample_bytree': population[i][6],
              'seed': 24}
        
        myXGBClassifier = XGBClassifier(**param).fit(X_train,y_train)
        #preds=myXGBClassifier.predict(X_Scaled)
        #preds = preds>0.5
        TestScore.append(round(myXGBClassifier.score(X_test,y_test), 4))
    return TestScore
###### / fonction de Fitness
    
######################################################################
#on utilise  ga_XGBClassifier - Attention cela dure un moment !!!!
###################################################################"

iterations=20  #nombre de générations
import ga_XGBClassifier #import de la classe
#on instancie un objet de classe ga_XGBClassifier
myGA = ga_XGBClassifier.ga_XGBClassifier(train_populationClassifier,
                                       nEpochs=iterations, X=X_Scaled, y=y,
                                       X_train=X_train, X_test=X_test, 
                                       y_train=y_train, y_test=y_test, 
                                       populationSize=10,
                                       crossover='un',nElite=4,tournamentOK=True)
myGA.runGA()  #on boucle dans l'objet créé

#Nouveau Fitness [0.8022, 0.8022, 0.8022, 0.8022, 0.7973, 0.8003, 0.8019, 0.795, 0.795, 0.798]

myGA.bestParams #meileurs parametres par génération
# dernière génération      [  0.14, 137.  ,  14.  ,   0.19,   0.6 ,   0.78,   0.56]])

myGA.bestfit   #meileur fitness par génération
nMax=myGA.bestfit.size





###########################################################  
#Graphique Evolution des meilleurs test  Scores
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
sns.lineplot(x=np.arange(0,nMax), y=myGA.bestfit[0:nMax])
fig.suptitle("Le réglage des paramètres permet d'améliorer le  test Score sensiblement", fontsize=14, fontweight='bold')
ax.set(xlabel='generation', ylabel='test  Score',
       title="Le Test score passe de "+"{0:.3f}".format(baseTestScore)+" à "+"{0:.3f}".format(myGA.bestfit[nMax-1]) )
ax.xaxis.set_ticks(range(nMax))
fig.text(.3,-.06,"Evolution des meilleurs Test Scores \n Recherche des meilleurs paramètres XGBoost", 
         fontsize=9)
#plt.show()
fig.savefig("QPPS6-GA-XGBClassifier-BestTestScores.png", bbox_inches="tight", dpi=600)


##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()











    

#!/usr/bin/env python
# coding: utf-8

# # PROJET

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


class READFILE:
    
    def __init__(self, file):
        self.file = file
        if file.endswith("csv"):
            self.read_csv()
        elif file.endswith("xlsx"):
            self.read_excel()

    def read_csv(self):
        return pd.read_csv(self.file)

    def read_excel(self):
        return pd.read_excel(self.file)
    
    def verif_presence(self):
        return os.path.exists(self.file)


df = READFILE("insurance_modified.csv") #BDD modifiée par l'auxiliaire
df = df.read_csv()


print(df.head())
print(f"Shape of data: {df.shape}") # formattage chaîne f-string


# In[3]:


# 1. Nettoyage des données

## 1.1 Plausibilité des données 

### Age
print(sorted(df['age'].unique()))#une donnée non plausible, soit le "64yo"
df.age = df.age.replace(['64yo'],64)
df['age'] = df['age'].astype(int)
df.loc[df['age'] == "64yo"]#donnée non plausible effacée


# In[4]:


#### Gestion des erreurs pour vérifier à nouveau que la valeur string "64yo" est effacée

def erreur_detect(column) :
    for index in range(0,len(column)):
        valeur = column[index]
        
        try:
            valeur/2 #si pas integer, float ou nan, va amener l'exception TypeError 

        except TypeError:
            print('La colonne contient des données non numériques')

        else:
            pass

        finally:
            print('execution finale') # j'aurais voulu que ça le fasse une seule fois après avoir tout vérifié, mais je n'ai
                                      # pas su comment. 
                    
erreur_detect(df.age) # On voit execution finale à chaque réitération de la boucle, indiquant qu'il n'y a aucune 
                      # valeur non numérique dedans.


# In[5]:


### Sexe
print(sorted(df['sex'].unique())) #une donnée non plausible, soit le "M" qui devrait être male
df['sex'] = df['sex'].replace(['M'],'male')
df.loc[df['sex'] == "M"] #donnée non plausible effacée


# In[6]:


### IMC
print(sorted(df['bmi'].unique())) # aucune donnée non plausible


# In[7]:


### Nombre d'enfants
print(sorted(df['children'].unique())) # aucune donnée non plausible


# In[8]:


### Fumeur
print(sorted(df['smoker'].unique())) # aucune donnée non plausible


# In[9]:


### Region
print(sorted(df['region'].unique()))# deux données anormales "Southwest" et "1"
df['region'] = df['region'].replace(['Southwest',"1"],['southwest',np.NaN])
df['region'].value_counts() #verification


# In[10]:


### Frais d'assurance
print(sorted(df['charges'].unique())) #aucune donnée non plausible


# In[11]:


## 1.2 Données manquantes

### Algorithme recherche linéaire pour savoir si on a des données manquantes ou pas
def linear_search(x, Table):
        
    if x in Table:
        
        reponse = True
    
    else:
        
        reponse = False
            
    return reponse


# In[12]:


linear_search(None, df.charges)# il y a des NaN dans charges et je ne sais pas pourquoi ça retourne false, mais j'ai essayé.


# In[13]:


### Pourcentage de données manquantes dans chaque colonne

df.isnull().mean() * 100 # 0.07% de données manquantes pour les colonnes bmi, region et charges, ce qui correspond à une donnée 
                         # manquante (1/1338). Très négligeable. Aucune donnée manquante pour les colonnes age, sex, children 
                         # et smoker.

# Après réflexion, la décision de imput les données manquantes a été changée. Je ne veux pas imput des valeurs 
# qui pourraient créer des mauvais exemples desquels le AI pourrait apprendre. Je préfère laisser cela sous forme NaN.


# In[14]:


## 1.3 Données extrêmes 

from scipy import stats

# On retire les données avec des valeurs z absolues supérieures à 3 
# colonne par colonne (les variables continues seulement, soit age, IMC et frais)

### Age
df = df[(np.abs(stats.zscore(df.age, nan_policy='omit')) < 3)]
df # 0 rangee retirée


# In[15]:


### IMC
df = df[(np.abs(stats.zscore(df.bmi, nan_policy='omit')) < 3)]
df # 5 rangées retirées


# In[16]:


### Frais
df= df[(np.abs(stats.zscore(df.charges, nan_policy='omit')) < 3)]
df # 9 rangees retirées, 14 en tout


# In[17]:


## 1.4 Normalité des données

# Ce code servira à vérifier la skewness et kurtosis de chaque colonne de variable numérique.

### Age
print(stats.skew(df['age']), stats.kurtosis(df['age'])) # skew = 0.05 et kurt = -1.25


# In[18]:


### IMC
print(stats.skew(df['bmi']), stats.kurtosis(df['bmi'])) # skew = 0.19 et kurt = -0.30


# In[19]:


### Nombre d'enfants
print(stats.skew(df['children']), stats.kurtosis(df['children'])) # skew = 0.74 et kurt = -0.58


# In[20]:


### Frais
print(stats.skew(df['charges']), stats.kurtosis(df['charges'])) # skew = 1.44 et kurt = 1.20

# Malgré le fait que certaines variables aient des indices de skewness ou de kurtosis en dehors de l'intervalle -1 à 1, selon 
# Curran, West, & Finch, 1996; des skewness entre +/- 2 et des  kurtosis entre +/-7 sont aussi acceptables, ce qui est notre
# cas ici et nous prendrons cet avis comme montré dans notre cours de statistiques 2.


# In[21]:


#Le tout pouvait être remplacé par la fonction anonyme suivante :
g = lambda col: print(col.name, stats.skew(col), stats.kurtosis(col))
print(g(df.age))
print(g(df.bmi))
print(g(df.children))
print(g(df.charges))


# In[22]:


# 2. Résolution des questions de recherche

## 2.1 Heatmap de corrélation

### Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
#### Sexe
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
#### Fumeur
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#### Region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)

corr = df.corr()
print(corr) # résultats
sns.heatmap(corr, cmap="Blues", annot=True) # présentation visuelle des résultats


# In[23]:


### Version plus complète avec les valeurs p de chaque corrélation et les corrélations significatives

import matplotlib.pyplot as pl

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def plot_cor_matrix(corr, mask=None):
    f, ax = pl.subplots(figsize=(11, 9))
    ax.set_title("Corrélations significatives")
    sns.heatmap(corr, ax=ax,
                mask=mask, 
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', linewidths=2, linecolor='black', cbar_kws={'orientation': 'horizontal'})


p_values = corr_sig(df)                    
print(p_values)                             # valeurs p des corrélations
mask = np.invert(np.tril(p_values<0.05))    # ne montre que les corrélations significatives
plot_cor_matrix(corr,mask)  

# Les corrélations significatives principales sont 
# smoker:charges (r=0.78, p<0.001)  
# age:charges (r=0.31, p<0.001) 
# bmi:charges (r=0.19, p<0.001) 
# bmi:region (r=0.15, p<0.001)
# age:bmi (r=0.12, p<0.001)


# In[24]:


## 2.2 Régression entre les frais et la variable ayant la corrélation la plus forte (fumeur)

import statsmodels
from statsmodels.formula.api import ols
model = ols("smoker ~ charges", df).fit()
print(model.summary()) # résultat, R^2 = 0.616 fort prédicteur

f = pl.figure(figsize=(14,6))
sns.violinplot(x='smoker', y='charges',data=df,palette='magma').set_title("Diagramme en violon des frais et de la consommation de tabac"); #Visualisation de la répartition fumeur:frais


# In[25]:


# Visualisations supplémentaires avec une variable aussi corrélée aux frais d'assurance : l'IMC

pl.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=df,palette='magma',hue='smoker')
ax.set_title('Nuage de points des frais et IMC')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=df, palette = 'magma', size = 8)

# Il semble que l'effet du bmi sur les frais est moindre en comparaison avec le fait de fumer ou pas.


# In[26]:


# 3. Apprentissage machine

## 3.1 Apprentissage supervisé (régression linéaire et forêts aléatoires)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

### Extraction de la matrice de fonctionnalités et du tableau cible
X_df = df.drop('charges', axis=1)
y_df = df['charges']

### Séparation en training dataset et test dataset

Xtrain,Xtest,y_train,y_test = train_test_split(X_df,y_df, random_state = 0)

### Méthode de régression linéaire
lr = LinearRegression().fit(Xtrain,y_train)

y_train_pred = lr.predict(Xtrain)
y_test_pred = lr.predict(Xtest)

print(lr.score(Xtest,y_test)) #résultat de régression relativement bon


# In[27]:


### Essayer avec un autre modèle pour un meilleur taux de prédiction (forêts aléatoires)
forest = RandomForestRegressor(n_estimators = 1000,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(Xtrain,y_train)
forest_train_pred = forest.predict(Xtrain)
forest_test_pred = forest.predict(Xtest)

print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred))) # Augmentation de 10% en taux de prédiction; credits au user Dandelion sur kaggle


# In[28]:


#### Visualisation des résultats de la méthode forêts aléatoire

pl.figure(figsize=(10,6))

pl.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Données entrainement')
pl.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Données test')
pl.xlabel('Valeurs prédites')
pl.ylabel('Décalage de la prédiction')
pl.title("Apprentissage supervisé : prédiction des frais d'assurance selon la méthode forêts aléatoires")
pl.legend(loc = 'upper right')
pl.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
pl.show()


# In[29]:


# 3. Apprentissage machine

## 3.1 Apprentissage non supervisé (ACP)

import sklearn
from sklearn.decomposition import PCA

# Par souci de simplicité, la valeur cible a été changée des frais à fumeur car nous n'avons vu que comment faire cela avec
# des variables catégorielles
X_df2 = df.drop('smoker', axis=1)
y_df2 = df['smoker']

pca = sklearn.decomposition.PCA(n_components=6)
pca.fit(X_df2)

#Composantes de la PCA et la variance expliquée
print(pca.components_)
print(pca.explained_variance_)


composantes_col = pca.transform(X_df2)
df['PCA_comp1'] = composantes_col[:, 0]
df['PCA_comp2'] = composantes_col[:, 1]
df['PCA_comp3'] = composantes_col[:, 2]
df['PCA_comp4'] = composantes_col[:, 3]
df['PCA_comp5'] = composantes_col[:, 4]
df['PCA_comp6'] = composantes_col[:, 5]

# Visualisation des résultats de classification de l'ACP
import seaborn as sns
sns.lmplot(x = "PCA_comp1", y = "PCA_comp2", hue='smoker', data=df, fit_reg=False).fig.suptitle("Apprentissage non supervisé PCA");
pl.show()


#Impression de la courbe ROC pour la PCA
import numpy as np
pl.plot(np.cumsum(pca.explained_variance_ratio_))
pl.title("Courbe ROC")
pl.xlabel('Nombre de composantes')
pl.ylabel('Variance expliquée cumulative');
pl.show()


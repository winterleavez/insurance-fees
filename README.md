# insurance-fees
Data Analysis Honors Class Project 

Bonjour,

Voici le code du projet final du cours PSY4016 Traitement de données en neurosciences cognitives
La base de données provient de Kaggle et s'intitule : Medical Cost Personal Datasets.

Le document insurance-modified.csv est la BDD modifiée par l'auxiliaire. PROJET.py est le code et ce texte consiste en un court rappel sur la BDD et un résumé des résultats obtenus en lien avec les questions de recherche établies dans le plan.

La banque de données choisie présentait les frais d’assurance maladie d’un échantillon de 1338 individus vivant dans un pays fictif. Pour prédire ces frais, la banque de données comportait aussi les variables suivantes : 

•	Âge du bénéficiaire principal
•	Sexe
•	IMC
•	Nombre d’enfants couverts par le régime d’assurance
•	Fumeur ou non
•	Région du pays où le bénéficiaire habite

La première question de recherche était de nature exploratoire et consistait à obtenir les différentes corrélations entre les frais d'assurance et le reste des variable. L'hypothèse était que les corrélations entre ces différentes variables et les frais soient positives et significatives.
De la matrice de corrélations effectuée, cette hypothèse est potentiellement confirmée (ou infirmée à vous de voir) :

  Les corrélations significatives les plus fortes sont fumeur:frais (r=0.78, p<0.001), age:frais (r=0.31, p<0.001) et imc:frais (r=0.19, p<0.001).

La seconde question de recherche était de déterminer à quel point la variable ayant eu la plus forte corrélation avec les frais dans Q1 prédisait ces derniers. L'hypothèse était que cette variable expliquerait plus de 20% de la variance des frais (R2>0.2). L'hypothèse a été confirmée. La régression frais:fumeur a eu comme résultat R2=0.616, faisait de la variable fumeur un très fort prédicteur des frais d'assurance.

En dernier lieu, des algorithmes d'apprentissage machine supervisés (forêts aléatoires) et non supervisés (PCA) ont été utilisés sur la base de données. La méthode forêts aléatoires était capable de prédire 81.8% de précision les frais d'assurance en prenant les autres variables. La méthode PCA a réussi à bien classifier les données de façon à distinguer les fumeurs des non fumeurs.

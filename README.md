# insurance-fees
Data Analysis Honors Class Project 

Bonjour,

Voici le code du projet final du cours PSY4016 Traitement de données en neurosciences cognitives
La base de données provient de Kaggle et s'intitule : Medical Cost Personal Datasets.

STEP 1
Le code principal se trouve dans le fichier insurance.py
The completed file calls all the functions it needs to perform de tasks in the different files.
To better understand what is happening, i invite you to go look at the files in
the code folder, for they contain all code used in this project along with certain
descriptions.

You will obtain a Regression result that is used to verify our first hypothesis

STEP 2
To visualise graphs for that analysis, first run the SQLdatabase file then run the Visualisation file

STEP 3 & 4
The movies were separated in 2 groups, old (2010 and before) and new (after 2010).
The goal of the supervised algorithms were to properly identify to wich the groups belong to.
2 supervised algorithm were chosen for that, respectively named AM_Gaussien and AM_KNN
Running those scripts will return both an accuracy score and a matrix table showcasing performance

STEP 5
The unsupervised algorithm that was chosen is in the file AM_ACP. Running it will return a graph showcasing the classified data.

# Système de recommandation de livre avec l'algorithme ALS (Filtrage Collaboratif) 

# HASSANI Samir & YAZID Tarik (M2 Big Data)


##  Pré-requis :
Python3 , Pyspark

- Pour installer Pyspark, suivre le lien :
	[https://towardsdatascience.com/installing-pyspark-with-java-8-on-ubuntu-18-04-6a9dea915b5b]


##  General Info :
- Le jeu de données de notre travail est à télécharger sur :
	[https://github.com/zygmuntz/goodbooks-10k]

##  Travail réalisé :
###  Chargement et traitement des données 
- Chargement du dataset "ratings.csv" qui contient 3 colonnes "user_id", "book_id" et "rating".
- Suppression des données dupliquées.
- Evaluation des données et suppression des livres ayant une moyenne de note inférieur à 3/5.
- Suppression des utilisateurs qui ont noté moins de 30 livres.
- Chargement du dataset de description des livres
	
### Entrainement et validation avec l'algorithme de recommandation ALS
- Recherche des meilleurs paramètres pour le modèle ALS.
- On prend les paramètres (rang, maxIter et regParam) qui généré le taux d'erreur (RMSE) le plus faible.
- Entrainement du modèle avec les paramètres obtenus précédemment sur les données d'apprentissage.
- Récupération de la recommandation de livre avec le modèle entrainé.

### Résultats 
- Récupération des déscriptions des livres recommandés à partir de "books.csv".
- Recommandation de livre pour un utilisateur choisi.
- Lancer une recherche sur le web des livres recommandés pour cet utilisateur, sur le site "Rakuten.fr".
	
	
## Commandes :
- Décompression du fichier "projet_als.tar.xz" avec la commande suivante :

	> tar -xvf project_als.tar.xz 
- Supression du dossier "projet_als.tar.xz" avec la commande suivante :

	> rm -r project_als.tar.xz
- Pour exécuter le projet, il faut se rendre dans le répértoire /projet_als avec un terminal puis lancer la commande :

	> python3 main.py
	 

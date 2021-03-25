import os
import sys

if __name__ == '__main__':

    from pyspark.sql import SparkSession
    import fonctions as f
    spark = SparkSession.builder.master("local").getOrCreate()

    # Chargement du dataset complet
    notes = "input/ratings.csv"
    data = f.read_file(notes)
    print(type(data))

    # Chargement du dataset de description des livres
    livres = "input/books.csv"
    books_data = f.read_file(livres)

    # Traitement de données
    # Suppression des livres ayant une moyenne de note inférieur à 3
    # Suppression des utilisateurs qui ont noté moins de 30 livres.
    # Pour avoir des données plus pertinentes pour un bon entrainement.
    data_final = f.traitement(data)
    data_final.show(5)
    data_final.count()

    # On dévise les données pour l'entrainement et le test
    (train, test) = data_final.randomSplit([0.7, 0.3])


    # Entrainement et validation avec l'algorithme de recommandation ALS 
    # Calcul du taux d'erreur
    # Recherche des meilleurs paramétres pour notre model ALS
    params = f.best_params(train, test)
    print(params)

    # Entrainement et validation avec l'algorithme de recommandation ALS avec les meilleurs paramètres trouvé dans l'étape précédente
    # Calcul du taux d'erreur
    predictions = f.train_predict(params, train, test)
    predictions.show(5)

    # Recommandation et affichage de livre pour un utilisateur donné  
    # Recherche des livres recommandés pour cet utilisateur sur le web (site : Rakuten)
    # Lancement des pages de recherche web 
    f.liste_recommandation(predictions, books_data)

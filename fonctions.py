from pyspark.sql import SparkSession
from pyspark.context import SparkContext as sc
import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pyspark
import numpy as np 
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS, ALSModel
import webbrowser

conf = pyspark.SparkConf()
sc = pyspark.SparkContext.getOrCreate(conf=conf)
sqlContext = SQLContext(sc)


spark = SparkSession.builder.master("local").getOrCreate()


def read_file(fichier):
    data = spark.read.options(header = True,inferSchema=True,delimiter=',').csv(fichier)
    # Suppression des données dupliquées
    data = data.dropDuplicates()
    data.show(10)
    data.printSchema()
    return data

def traitement(data):
    sqlContext.registerDataFrameAsTable(data, "table1")
    df2 = sqlContext.sql("SELECT user_id, book_id, rating from table1")
    df2.collect()

    ## Suppression des livres ayant une moyenne de note inférieur à 3
    df3 = df2.groupBy("book_id").avg("rating")
    df3.collect()
    df3 = df3.withColumnRenamed("book_id", "b_id")

    dataset1 = data.join(df3, data.book_id == df3.b_id)
    dt = dataset1.withColumnRenamed("avg(rating)", "moyenne")

    dat = dt.where("moyenne > 3")

    # Suppression des utilisateurs qui ont noté moins de 30 livres.
    user_30 = dat.groupBy("user_id").count()
    user_30 = user_30.withColumnRenamed("user_id", "u_id")


    dataset = user_30.join(dat, user_30.u_id == dat.user_id)
    d = dataset.where("count > 30")
    
    data_final = d.select("user_id", "book_id","rating")
    return data_final

def best_params(train, test):
    # Recherche des meilleurs paramétres pour notre model ALS
    rangs = [9,10]
    maxIter = [10,20]
    regParam = [ 0.1, 0.01]
    min_erreur = float('inf')
    best_rang = -1
    best_iter = 0
    best_reg = 0
    tab = []
    for ran in rangs:
        for x in maxIter:
            for y in regParam:
                        
                als = ALS(rank=ran, maxIter=x, regParam=y,
                    userCol="user_id", itemCol="book_id", ratingCol="rating")    
                print('Training ... ')
                model = als.fit(train)
                print("Prediction ... ")
                predictions = model.transform(test)
                predict= predictions.filter(col('prediction')!=np.nan)
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
                print("Error calculation ...")
                rmse = evaluator.evaluate(predict)
                print("Rang :" ,(ran)," maxIter :",x," regParam :", y,"  Root-Mean-Square Error = " + str(rmse), "\n")
                if rmse < min_erreur:
                    min_erreur = rmse
                    best_rang = ran
                    best_iter = x
                    best_reg = y
    tab.append(best_rang)
    tab.append(best_iter)
    tab.append(best_reg)
    print(tab)
    print("\n")    
    print("Best rang = ", best_rang, "\n")
    print("Best maxIter =  ", best_iter, "\n")
    print("Best regParam = ", best_reg, "\n")
    return tab

def train_predict(tab, train, test):
    als = ALS(rank=tab[0], maxIter = tab[1], regParam=tab[2], userCol="user_id", itemCol="book_id", ratingCol="rating")    
    print('Training ... ')
    model = als.fit(train)
    print("Prediction ... ")
    predictions = model.transform(test)
    predict= predictions.filter(col('prediction')!=np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    print("Error calculation ...")
    rmse = evaluator.evaluate(predict)
    print("Root-Mean-Square Error = " + str(rmse))
    predictions.show(10)
    return predictions

def liste_recommandation(predictions, books_data):
    # Liste des recommandations de livre
    predictions.join(books_data, "book_id").select("user_id", "authors", "original_title", "original_publication_year").show(10)

    # Recommandation de livres pour un utilisateur donné
    for_one_user = predictions.filter(col("user_id")==42).join(books_data, "book_id").select("user_id", "book_id","original_title","authors")
    for_one_user.show(5)

    # Recherche des deux premiers livre recommandé à un utilisateur sur le web (site: Rakuten)
    link = "https://fr.shopping.rakuten.com/search/"
    for book in for_one_user.take(2):
        bookURL = link + str(book.original_title)
        print(book.original_title)
        webbrowser.open(bookURL)
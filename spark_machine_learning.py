import sys
import os
from pyspark import *
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import zipfile
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
import pandas as pd
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession \
             .builder \
             .master("local[*]") \
             .getOrCreate()

# filePath = """sf-airbnb-clean-100p.parquet/"""

# Path to the file
filepath = 'data/sf-airbnb-clean-100p.parquet'

# print(df.info())
# print(df.head)
airbnbDF = spark.read \
                .parquet(filepath)

# airbnbDF = spark.read.parquet("/Users/oluwadaraadedeji/Desktop/Spring 2024/AirBnBpricePrediction/sf-airbnb-clean-100p.parquet")
# print(airbnbDF.printSchema())
#To get schema for programming
print(airbnbDF.schema)
airbnbDF \
.select("neighbourhood_cleansed"
         , "room_type"
         , "bedrooms"
         ,"bathrooms"
         ,"number_of_reviews"
         , "price").show(5)

#Repartition to improve availability 
airbnbDF = airbnbDF.repartition(200)

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)
print(f"""There are {trainDF.count()} rows in the training set, and {testDF.count()} in the test set""")
#Vectorizer
vecAssembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
vecTrainDF = vecAssembler.transform(trainDF)
vecTrainDF.select("bedrooms", "features", "price").show(10)
#Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="price")
lrModel = lr.fit(vecTrainDF)

m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept, 2)
print(f"""The formula for the linear regression line is price = {m}*bedrooms + {b}""")

pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF  \
   .select("bedrooms" \
           ,"features" \
           , "price"  \
           , "prediction").show(10)

regressionEvaluator = RegressionEvaluator(
                          predictionCol="prediction",
                          labelCol="price",
                          metricName="rmse")
rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE for linear regression is {rmse:.1f}")


# Create an instance of VectorAssembler with multiple input columns
vecAssembler = VectorAssembler(
    inputCols=["accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights", "review_scores_rating"],
    outputCol="features"
)

# Transform the DataFrame to include a new 'features' column that contains vectors of the input features
vecTrainDF = vecAssembler.transform(trainDF)

#Multiple linear regression
# Show the first 10 rows of the DataFrame to verify the transformation, displaying the specified columns and the new 'features' column
vecTrainDF \
    .select("accommodates" \
           , "bathrooms" \
           , "bedrooms" \
           , "beds" \
           , "minimum_nights" \
           , "review_scores_rating" \
           , "features" \
           , "price").show(10)

lr = LinearRegression(featuresCol="features" \
                      , labelCol="price")
lrModel = lr.fit(vecTrainDF)

X1 = round(lrModel.coefficients[0], 2)
X2 = round(lrModel.coefficients[1], 2)
X3 = round(lrModel.coefficients[2], 2)
X4 = round(lrModel.coefficients[3], 2)
X5 = round(lrModel.coefficients[4], 2)
X6 = round(lrModel.coefficients[5], 2)
c = round(lrModel.intercept, 2)
print(f"""The formula for the linear regression line is price = {X1}*accommodates + {X2}bathrooms +{X3}bedrooms +{X4}beds +{X5}minimum_nights+{X6}review_scores_rating+ {c}""")


pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("accommodates" \
              , "bathrooms" \
              , "bedrooms"  \
              , "beds" \
              , "minimum_nights" \
              , "review_scores_rating" \
              , "features" \
              , "price" \
              , "prediction").show(10)


regressionEvaluator = RegressionEvaluator(
                          predictionCol="prediction",
                          labelCol="price",
                          metricName="rmse")
rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE for linear regression with multiple independent variables is {rmse:.1f}")

#RandomForest
rf = RandomForestRegressor(featuresCol="features" \
                           , labelCol="price")

# Fit the RandomForestRegressor model to the training data
rfModel = rf.fit(vecTrainDF)

pipeline = Pipeline(stages=[vecAssembler, rf])
pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("accommodates" \
               , "bathrooms"  \
              , "bedrooms"  \
              , "beds" \
              , "minimum_nights" \
              , "review_scores_rating" \
              , "features" \
              , "price" \
              , "prediction").show(10)


regressionEvaluator = RegressionEvaluator(
                          predictionCol="prediction",
                          labelCol="price",
                          metricName="rmse")
rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE for Random Forest is {rmse:.1f}")

# # Hyperparametertuning
# rf = RandomForestRegressor(featuresCol="features", labelCol="price")

# # Define ParamGrid
# paramGrid = ParamGridBuilder() \
#     .addGrid(rf.maxDepth, [5, 10, 15]) \
#     .addGrid(rf.numTrees, [10, 20, 30]) \
#     .addGrid(rf.maxBins, [20, 30, 40]) \
#     .build()

# # Create CrossValidator
# evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
# crossval = CrossValidator(estimator=rf,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=evaluator,
#                           numFolds=3)
# print(vecTrainDF.select("accommodates" \
#                          , "bathrooms" \
#                          , "bedrooms"  \
#                          , "beds"  \
#                          , "minimum_nights" \
#                          , "review_scores_rating" \
#                          , "features" \
#                          , "price").show(10))
# # Fit CrossValidator

# cvModel = crossval.fit(vecTrainDF)

# # # Get best model
# bestModel = cvModel.bestModel

# # Define XGBoostRegressor with featuresCol and labelCol
# xgb = SparkXGBRegressor(
#   features_col="features",
#   label_col="price",
#   num_workers=2
# )

# # Fit the XGBoostRegressor model to the training data
# xgbModel = xgb.fit(vecTrainDF)

# pipeline = Pipeline(stages=[vecAssembler, xgb])
# pipelineModel = pipeline.fit(trainDF)
# predDF = pipelineModel.transform(testDF)
# predDF.select("accommodates"  \
#               , "bathrooms"  \
#               , "bedrooms"  \
#               , "beds" \
#               , "minimum_nights" \
#               , "review_scores_rating" \
#               , "features" \
#               , "price" \
#               , "prediction").show(10)


# regressionEvaluator = RegressionEvaluator(
#                           predictionCol="prediction",
#                           labelCol="price",
#                           metricName="rmse")
# rmse = regressionEvaluator.evaluate(predDF)
# print(f"RMSE for Xgboost is {rmse:.1f}")

print("Completed")
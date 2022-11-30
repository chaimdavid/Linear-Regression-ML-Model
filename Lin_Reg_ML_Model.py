#Importing needed modules
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
spark=SparkSession.builder.appName('cruise').getOrCreate()
#Importing the data into a Spark dataframe
data=spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)
#Indexing the contents of a column that are strings so that it can be used as a feature from the model
indexer=StringIndexer(inputCol='Cruise_line',outputCol='lineIndex')
indexed=indexer.fit(data).transform(data)
indexed.columns
#Creating a new column that contains all the features as vectors per row
assembler=VectorAssembler(inputCols=['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density','lineIndex'],outputCol='features')
output=assembler.transform(indexed)
output.select('features').show()
final_data=output.select('features','crew')
#Splitting the original data into two sub-dataframes. A training one with the 70% of the original data and a test one with the remaining 30%
train_data,test_data=final_data.randomSplit([0.7,0.3])
#Creation of the model
lr = LinearRegression(labelCol='crew')
lrModel = lr.fit(train_data,)
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
#Evaluation of the model by making use of the untouched test data
test_results = lrModel.evaluate(test_data)
test_results.residuals.show()
#Creation of the model's predictions
unlabeled_data = test_data.select('features')
predictions = lrModel.transform(unlabeled_data)
predictions.show()
#Printing of two statistical evaluation measures
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
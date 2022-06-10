# Databricks notebook source
# MAGIC %md
# MAGIC Prework: Create the datasets required for this demo

# COMMAND ----------

# MAGIC %run ./retail-churn-data-prep

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Predicting Churn with Databricks ML
# MAGIC 
# MAGIC 
# MAGIC <div><img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/retail-ml-pipeline.png"/></div>
# MAGIC 
# MAGIC *Note: this demo is designed to showcase Databricks ML capabilities and uses sythetic data; real-world scenarios will require a more comprehensive approach*
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fretail%ml%2Fproduct_classification%2Fml_product_classification_02&dt=ML">
# MAGIC <!-- [metadata={"description":"Create a model to classify product.<br/><i>Usage: basic MLFlow demo, auto-ml, feature store.</i>",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Science", "components": ["auto-ml", "mlflow", "feature-store"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0) Prepare Environment

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql.functions import struct
import pyspark.sql.functions as F
from pyspark.sql.functions import col
  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from databricks.feature_store import FeatureStoreClient, FeatureLookup
from databricks import automl

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip freeze

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1) Explore dataset

# COMMAND ----------

#Here we are reading a csv from the cloud file system.  
#You can also bring in data from other sources with connectors, e.g jdbc: https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/sql-databases

churn_df_pyspark = spark.read \
                 .format("csv") \
                 .option("header", True) \
                 .option("inferSchema", True) \
                 .load("dbfs:/tmp/mimiq/retail/customer_churn.csv")
    
display(churn_df_pyspark)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basic feature transforms

# COMMAND ----------

# 0/1 -> boolean
churn_df_pyspark = churn_df_pyspark.withColumn("senior_citizen", F.col("senior_citizen") == 1)

# Yes/No -> boolean
for yes_no_col in ["loyalty_member", "paperless_billing", "item_protection", "churn"]:
  churn_df_pyspark = churn_df_pyspark.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")

churn_df = churn_df_pyspark.toPandas()
display(churn_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Split datasets

# COMMAND ----------

train, test = train_test_split(churn_df, random_state=123)
X_train = train.drop(["churn", "customerID"], axis=1)
X_test = test.drop(["churn", "customerID"], axis=1)
y_train = train.churn
y_test = test.churn

# Save test set to hive metastore for batch prediction
sqlContext.sql('drop table mimi_qunell_databricks_com.churn_test_set')
spark.createDataFrame(X_test).write.mode("overwrite").saveAsTable("mimi_qunell_databricks_com.churn_test_set")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) Train a Random Forest Classifier
# MAGIC MlFlow will automatically log data from this experiment!

# COMMAND ----------

mlflow.sklearn.autolog(silent=True)
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 20
  max_depth = 20
  max_features = 5
  
  # Create and train model.
  rf_classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features, random_state=3)
  
  # Define One-Hot encoding for categorical variables
  categorical_features_to_encode = X_train.columns[X_train.dtypes==object].tolist()
  col_transform = make_column_transformer((OneHotEncoder(), categorical_features_to_encode), remainder = "passthrough")
  
  # Define pipeline, fit, and predict
  pipeline = make_pipeline(col_transform, rf_classifier)
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)

  # Basic Evaluation
  accuracy_score(y_test, y_pred)
  print(f"The accuracy of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
  plot_confusion_matrix(pipeline, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) Streamline with AutoML
# MAGIC 
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient! 
# MAGIC 
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC 
# MAGIC 
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>

# COMMAND ----------

model_r=automl.classify(churn_df.drop(['customerID'], axis=1), target_col='churn', timeout_minutes=5)

# COMMAND ----------

#Code to programmatically return the best model from a set of AutoML experiments
print(model_r.best_trial)

# COMMAND ----------

#Code to register the best model in the model registry
mlflow.register_model(model_r.best_trial.model_path, "mimiq_retail_churn")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 5) (Console) Demonstrate MLFlow Model Registry
# MAGIC This could be done via python API as well -- pick your favorite
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>
# MAGIC <br>
# MAGIC All our training information are now saved in MLFLow and available in the MLFLow side-bar and the UI:
# MAGIC 
# MAGIC - Model dependencies
# MAGIC - Hyper-parameters
# MAGIC - Metrics and artifacts (custom images, confusion matrix etc)
# MAGIC - The model itself, automatically serialized
# MAGIC 
# MAGIC This guarantee reproducibility and tracking over time, improving team efficiency and model governance.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Demonstrate Batch Prediction on Test Set

# COMMAND ----------

model_name = "mimiq_retail_churn"
model_uri = f"models:/{model_name}/Production"

# create spark user-defined function for model prediction
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="string")

input_table_name = "mimi_qunell_databricks_com.churn_test_set"
table = spark.table(input_table_name)
output_df = table.withColumn("prediction", predict(struct(*table.columns)))
display(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Show what is tracked by MLflow programmatically

# COMMAND ----------

df = spark.read.format("mlflow-experiment").load()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6) Simplify & Strengthen via the Feature Store
# MAGIC 
# MAGIC With large datasets, it can be common to create many many derived features - windowed aggregates, one-hot encoding, complex mathematical formulas, geospatial functions/enrichment, etc. It can also be common to rename columns and drop/fill nulls. 
# MAGIC 
# MAGIC These transforms aren't necessarily going to be done (nor should they be) in the data engineering pipeline itself.  They are specific to the ML modeler's needs.<br> 
# MAGIC 
# MAGIC __The Feature store is a vital piece in model governance:__
# MAGIC 
# MAGIC 1. Feature Discoverability and Reusability across the entire organization (build feature once, reuse multiple time)<br>
# MAGIC 2. Feature versioning<br>
# MAGIC 3. Upstream and downstream Lineage (where is the feature coming from, which models are using it)<br>
# MAGIC 4. Ensure your model will use the same data for training and inferences<br>

# COMMAND ----------

fs_cols = ["customerID", "gender", "senior_citizen", "years_as_customer", "loyalty_member", "item_protection", "paperless_billing", "total_spend"]

customer_fs_df = churn_df_pyspark[fs_cols]
display(customer_fs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Write the customer data to the feature store.

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS mimi_qunell_databricks_com.retail_fs;
# MAGIC DROP TABLE IF EXISTS mimi_qunell_databricks_com.retail_fs1;
# MAGIC --also delete from FS UI

# COMMAND ----------

fs = FeatureStoreClient()

fs_cols = ["customerID", "gender", "senior_citizen", "years_as_customer", "loyalty_member", "item_protection", "paperless_billing", "total_spend"]
customer_fs_df = churn_df_pyspark[fs_cols]

customer_features_table = fs.create_table(
  name='mimi_qunell_databricks_com.retail_fs',
  primary_keys='customerID',
  schema=customer_fs_df.schema,
  description='Retail customer details')

fs.write_table("mimi_qunell_databricks_com.retail_fs", customer_fs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Browsing the Feature Store
# MAGIC 
# MAGIC The tables are now visible and searchable in the [Feature Store](/#feature-store/jpzebrowski.retail_fs) -- try it!

# COMMAND ----------

### Generic model
def train_rf_classifier(training_set):
  
  # Split into training, testing
  training_pd = training_set.load_df().toPandas()
  X = training_pd.drop("churn", axis=1)
  y = training_pd["churn"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  
  n_estimators = 20
  max_depth = 20
  max_features = 5
  
  # Create and train model.
  rf_classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features, random_state=3)
  
  # Define One-Hot encoding for categorical variables
  categorical_features_to_encode = X_train.columns[X_train.dtypes==object].tolist()
  col_transform = make_column_transformer((OneHotEncoder(), categorical_features_to_encode), remainder = "passthrough")
  
  # Define pipeline, fit, and predict
  pipeline = make_pipeline(col_transform, rf_classifier)
  pipeline_model = pipeline.fit(X_train, y_train)
  
  return pipeline_model, X, y

# COMMAND ----------

# MAGIC %md
# MAGIC #### Leverage Feature Store for training!
# MAGIC Now, you only need the identifier and label... everything else is looked up

# COMMAND ----------

customer_churn_df = churn_df_pyspark[["customerID", "churn"]]
display(customer_churn_df)

# COMMAND ----------

mlflow.autolog(log_input_examples=True, silent=True)

with mlflow.start_run():    
  # Providing model_feature_lookups in order to create training set! 
  model_feature_lookups = [FeatureLookup(table_name = customer_features_table.name, lookup_key = 'customerID')]
  training_set = fs.create_training_set(customer_churn_df, model_feature_lookups, label="churn", exclude_columns="customerID")
  
  # Train model
  pipeline_model, X, y = train_rf_classifier(training_set)

  # This will package the model together with metadata about feature lookups
  fs.log_model(
    model = pipeline_model,
    artifact_path = "model",
    flavor = mlflow.sklearn,
    training_set = training_set,
    registered_model_name = "mimiq_fs_retail_churn",
    input_example = X[:100],
    signature = infer_signature(X, y)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Predict using feature store
# MAGIC Here, only pass in `customerID` and the model identifier. Why should we need anything else?

# COMMAND ----------

batch_input_df = customer_churn_df.select("customerID")
display(batch_input_df)

# COMMAND ----------

# Use the feature store augment your input data
predictions = fs.score_batch("models:/mimiq_fs_retail_churn/1", batch_input_df, result_type='string')
display(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC #More To Discuss
# MAGIC - Online feature store for high concurrency real-time predictions: https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store/concepts#feature-table
# MAGIC - Webhooks - to enable workflows for model creation and state transitions: https://databricks.com/blog/2022/02/01/streamline-mlops-with-mlflow-model-registry-webhooks.html  and  https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry-webhooks
# MAGIC - Automated Jobs, Orchestration: https://databricks.com/blog/2021/11/01/now-generally-available-simple-data-and-machine-learning-pipelines-with-job-orchestration.html
# MAGIC - JDBC/ODBC connections to external databases: https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/sql-databases
# MAGIC - 2020 DAIS (Sean Owen) Introducing MLflow for End-to-End Machine Learning on Databricks:  https://www.youtube.com/watch?v=nx3yFzx_nHI
# MAGIC - 2021 DAIS (Rafi Kurlansik) Learn to Use Databricks for the Full ML Lifecycle: https://www.youtube.com/watch?v=dQD2gVPJggQ&t=565s
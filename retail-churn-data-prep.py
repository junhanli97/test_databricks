# Databricks notebook source
# MAGIC  %sh wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

dbfs_path = 'dbfs:/tmp/mimiq/feature-store/Telco-Customer-Churn.csv'
dbutils.fs.cp('file:/databricks/driver/Telco-Customer-Churn.csv', dbfs_path)


# COMMAND ----------

churn_df = spark.read \
                 .format("csv") \
                 .option("header", True) \
                 .option("inferSchema", True) \
                 .load(dbfs_path)
    
display(churn_df)

churn_df.createOrReplaceTempView("churn_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE mimi_qunell_databricks_com.retail_churn AS (
# MAGIC SELECT customerID, gender, SeniorCitizen AS senior_citizen, tenure AS years_as_customer, PhoneService AS loyalty_member, DeviceProtection AS item_protection, PaperlessBilling AS paperless_billing, FLOOR(RAND()*10000) AS total_spend, churn FROM churn_view)

# COMMAND ----------

tmp_churn_df = sqlContext.sql("SELECT * FROM mimi_qunell_databricks_com.retail_churn")

tmp_churn_df.write.format("csv") \
        .mode('overwrite') \
        .option("header", True) \
        .save('dbfs:/tmp/mimiq/retail/customer_churn.csv')

# COMMAND ----------


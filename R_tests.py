import pandas as pd
from R.convert_to_r_time_series import convert_to_r_time_series
from rpy2 import robjects
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

#
# robjects.r.source("R/auto_arima.R")
#
# train_data = convert_to_r_time_series(pd.read_csv("data/CT_train.csv", index_col="time")["demand"][0:100], frequency=24)
# test_data = convert_to_r_time_series(pd.read_csv("data/CT_test.csv", index_col="time")["demand"], frequency=24)
# #
# r_auto_arima = robjects.r["auto_arima"]# Call the R function to get the forecast values
# r_forecast_values = r_auto_arima(train_data, test_data)
#
# # Convert the R forecast values back to a pandas Series
# forecast_values = pd.Series(list(r_forecast_values.rx2('mean')), index=test_data.index)



from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ARIMAForecasting").getOrCreate()
# spark.sparkContext.setLogLevel("INFO")

# Load data into Spark DataFrame
data = spark.read.csv("data/CT_test.csv", header=True, inferSchema=True)

# Convert data["demand"] column to Pandas Series
demand_series = data.select(col("demand")).toPandas()["demand"]

# Convert data["time"] column to Pandas Series
time_series = data.select(col("time").cast("string")).toPandas()["time"]

train_data = convert_to_r_time_series(demand_series, time_series, frequency=12
                                      )


print(data)

# # Define the forecasting function
# def perform_forecast(chunk):
#     # Convert chunk to R time series using rpy2
#     r_time_series = convert_to_r_time_series(chunk, frequency=24)
#
#     # Perform ARIMA forecasting using rpy2
#     r_forecast = r_auto_arima(r_time_series)
#
#     return r_forecast
#
# # Split data into chunks and apply forecasting function in parallel
# forecasted_chunks = data.rdd.mapPartitions(perform_forecast)
#
# # Convert forecasted chunks back to DataFrame
# forecasted_data = forecasted_chunks.toDF()
#
# # Save the forecasted data
# forecasted_data.write.mode("overwrite").csv("forecasted_results")

# Stop Spark session
spark.stop()

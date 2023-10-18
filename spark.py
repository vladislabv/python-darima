from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from py_handlers.converters import convert_to_r_time_series
import findspark
from rpy2 import robjects
from pyspark.sql.functions import col
findspark.init()
# from pyspark.sql.functions import sum
import pandas as pd


conf = SparkConf().setAppName("DarimaModel")
# conf.set("spark.pyspark.python", "/venv/Scripts/python")
# conf.set("spark.pyspark.driver.python", "/venv/Scripts/python")
spark = SparkSession.builder.master("local[1]").config(conf=conf).getOrCreate()

def arima_modeling(iter):
    rows = list(iter)
    demand_values = [row["demand"] for row in rows]
    time_values = [str(row["time"]) for row in rows]
    ts = convert_to_r_time_series(demand_values, time_values, frequency=24)
    forecasted_values = auto_arima(ts)
    yield forecasted_values

def auto_arima(ts):
    robjects.r.source("../R/auto_arima.R")
    r_auto_arima = robjects.r["auto_arima"]
    r_forecast_arima = robjects.r["forecast_arima"]
    arima_model = r_auto_arima(ts)
    forecasted_values = r_forecast_arima(arima_model, ts)
    return forecasted_values

data = spark.read.csv("data/CT_test.csv", header=True, inferSchema=True)
parts = data.repartition(4).rdd
converted_ts_rdd = parts.mapPartitions(arima_modeling)


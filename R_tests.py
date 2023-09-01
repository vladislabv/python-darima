from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from R.convert_to_r_time_series import convert_to_r_time_series
import findspark
from rpy2 import robjects
from pyspark.sql.functions import col
findspark.init()
# from pyspark.sql.functions import sum
import pandas as pd
import matplotlib.pyplot as plt

def forecast_arima(ts):
    robjects.r.source("R/auto_arima.R")
    r_auto_arima = robjects.r["auto_arima"]
    r_forecast_arima = robjects.r["forecast_arima"]
    arima_model = r_auto_arima(ts)
    return r_forecast_arima(arima_model, ts)

data = pd.read_csv("data/CT_test.csv")

demand_values = data["demand"].tolist()
time_values = data["time"].tolist()

ts = convert_to_r_time_series(demand_values, time_values, frequency=24)
forecasted_values = forecast_arima(ts)

values = pd.read_csv("forecasted_values.csv")
forecasted = values["Point.Forecast"]


plt.scatter(time_values[0:100], demand_values[0:100], color="green")
plt.scatter(time_values[0:100], forecasted[0:100], color="red")
plt.show()
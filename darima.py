"""
darima.py
~~~~~~~~~~

This Python module contains an example Apache Spark ETL job definition
that implements best practices for production ETL jobs. It can be
submitted to a Spark cluster (or locally) using the 'spark-submit'
command found in the '/bin' directory of all Spark distributions
(necessary for running any Spark job, locally or otherwise). For
example, this example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master spark://localhost:7077 \
    --py-files packages.zip \
    --files configs/etl_config.json \
    jobs/etl_job.py

where packages.zip contains Python modules required by ETL job (in
this example it contains a class to provide access to Spark's logger),
which need to be made available to each executor process on every node
in the cluster; etl_config.json is a text file sent to the cluster,
containing a JSON object with all of the configuration parameters
required by the ETL job; and, etl_job.py contains the Spark application
to be executed by a driver process on the Spark master node.

For more details on submitting Spark applications, please see here:
http://spark.apache.org/docs/latest/submitting-applications.html

Our chosen approach for structuring jobs is to separate the individual
'units' of ETL - the Extract, Transform and Load parts - into dedicated
functions, such that the key Transform steps can be covered by tests
and jobs or called from within another environment (e.g. a Jupyter or
Zeppelin notebook).
"""

# External Packages
import time
import json

import pandas as pd
import numpy as np
from rpy2 import robjects
from py_spark.spark import start_spark

# Internal Packages
from py_handlers.converters import convert_to_r_time_series, rvector_to_list_of_tuples, convert_result_to_df
from py_handlers.utils import ppf


class Darima:
    """
    Darima is the Meta Class for modelling.

    """

    def __init__(self, datapath: str = "data/CT_test.csv",
                 column_name_value: str = "demand", column_name_time: str = "time"):
        """
        Initializing the datainput, columnnames and datalocation.

        :param datapath: path to the data should be .csv format
        :type datapath: str
        :param column_name_value: The name of the Values column
        :type column_name_value: str
        :param column_name_time: The name of the time column
        :type column_name_time: str
        """

        with open("configs/darima_config.json", 'r') as config_file:
            self.config_darima = json.load(config_file)
        self.datapath = datapath
        self.num_of_partitions = self.config_darima['num_partitions']
        self.frequency = self.config_darima['data_time_freq']
        self.column_name_value = column_name_value
        self.column_name_time = column_name_time

    def darima(self):
        """
        Creating a spark Session named DarimaModel.

        Calling values from darima_config.json.

        Main darima method. Calling all needed methods to get the coefficients.

        Map-Reduce step is done here.

        :return: None
        """

        # start Spark application and get Spark session, logger and config
        spark, log, config = start_spark(
            app_name='DarimaModel',
            files=[]
        )

        # log that main ETL job is starting
        log.warn('Darima job is up-and-running')

        # execute ETL (Darima) pipeline with Map and Reduce steps
        data = self.extract_data(spark)
        data_transformed = self.mapreduce_transform_data(data).collect()
        df_ar, df_sigma, df_beta = convert_result_to_df(data_transformed)
        print(df_ar, df_sigma, df_beta)

        # load_data(data_transformed)

        # log the success and terminate Spark application
        log.warn('Darima is finished')
        time.sleep(1000)
        spark.stop()
        return None

    def extract_data(self, spark):
        """
        Load data from CSV file format.

        :param spark: Spark session object.
        :return: Spark DataFrame.
        """
        df = (
            spark
            .read
            .csv(self.datapath, header=True, inferSchema=True)
        )

        return df

    def mapreduce_transform_data(self, df, dlsa: bool = False):
        """
        Transform original dataset.

        :param df: Input DataFrame.
        :return: Transformed DataFrame.
        """
        parts = (
            df
            .repartition(self.num_of_partitions)
            .rdd
        )

        converted_ts_rdd = (
            parts
            .mapPartitions(lambda x: self.map_arima(x))
            .flatMap(lambda x: x)
        )
       
        if dlsa:
            # Define the initial value for each key
            initial_value = 0

            # First lambda expression for Within-Partition Reduction Step::
            # a: is the current sum for the key.
            # b: is a SCALAR that holds the next Value
            calc_within_parts = lambda a, b: a + b

            # Second lambda expression for Cross-Partition Reduction Step::
            # a: is the running sum for the key.
            # b: is the next partition's sum.
            calc_cross_parts = lambda a, b: a + b

            sums_over_keys = converted_ts_rdd.aggregateByKey(initial_value, calc_within_parts, calc_cross_parts)
            # Collect the results
            result = pd.DataFrame(sums_over_keys.collect(), columns=["coef", "value"])

            # Generate diag according Sig_inv_sum_value
            #--------------------------------------
            # Sig_inv_sum_inv = 1/Sig_inv_sum_value * np.identity(p) # p-by-p

            # Get Theta_tilde and Sig_tilde
            #--------------------------------------
            # Theta_tilde = Sig_inv_sum_inv.dot(Sig_invMcoef_sum) # p-by-1
            # Sig_tilde = Sig_inv_sum_inv*sample_size # p-by-p
        else:
            # holds initial values for sum and count
            initial_value = (0, 0)

            calc_within_parts = lambda a, b: (a[0] + b, a[1] + 1)

            calc_cross_parts = lambda a, b: (a[0] + b[0], a[1] + b[1])
            mean_coeffs = converted_ts_rdd.aggregateByKey(initial_value, calc_within_parts, calc_cross_parts)

            # Finally, calculate the average for each KEY, and collect results.
            result = mean_coeffs.mapValues(lambda v: v[0] / v[1])
        
        return result

    def map_arima(self, iterator):
        """
        Will go through a partition (iterator)  convert it into an R-TimeSeries and yield the coefficients.

        :param iterator: A partion of a spark RDD
        :type iterator: spark RDD format
        :param frequency: See documentation of convert_to_r_time_series (4, 12, 24, 60, 3600)
        :type frequency: int
        :return: Will return a list of tuples with all coefficients
        :rtype: list(tuple())
        """
        rows = list(iterator)
        demand_values = [row["demand"] for row in rows]
        time_values = [str(row["time"]) for row in rows]
        ts = convert_to_r_time_series(demand_values, time_values, self.frequency)
        result = self.auto_arima(ts)
        yield result

    def auto_arima(self, ts):
        """
        Calling the R-Package within this project.

        Working with R-Objects.

        Is calling the function auto_arima within the R-Package.

        Converting values into a valid Python object.

        :param ts: TimeSeries
        :type ts: Should be R-TimeSeries
        :return: Will return List of tuples with named coefficients
        :rtype: list(tuple())
        """
        robjects.r.source("R/auto_arima.R")
        r_auto_arima = robjects.r["auto_arima"]
        apply_dlsa = self.config_darima.get('dlsa', False)
        arima_model_coefficients = r_auto_arima(ts, apply_dlsa)
        return rvector_to_list_of_tuples(arima_model_coefficients)

    def predict_ar(self, Theta, sigma2, x, n_ahead=1, se_fit=True):
        # Check arguments
        if n_ahead < 1:
            raise ValueError("'n_ahead' must be at least 1")
        if x is None:
            raise ValueError("Argument 'x' is missing")
        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a NumPy array")

        h = n_ahead
        n = len(x)
        st = x.index[1]  # Assuming x is a pandas Series with DateTimeIndex
        coef = Theta
        p = len(coef) - 2  # AR order
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)] + [x.shift(i) for i in range(1, p + 1)])

        # Fitted values
        fits = np.dot(X, coef).flatten()
        res = x - fits

        # Forecasts
        y = np.append(x, np.zeros(h))
        for i in range(h):
            y[n + i] = np.sum(coef * np.concatenate(([1, n + i], y[n + i - np.arange(1, p + 1)])))
        pred = y[n + np.arange(h)]

        # Standard errors
        if se_fit:
            psi = np.polymul(np.array([1]), coef[-p:])
            vars = np.cumsum(np.polymul(np.array([1]), psi ** 2))
            se = np.sqrt(sigma2 * vars[-h:])
            result = {"fitted": fits, "residuals": res, "pred": pred, "se": se}
        else:
            result = {"fitted": fits, "residuals": res, "pred": pred}

        return result
    
    def forecast_darima(self, Theta, sigma2, x, period, h=1, level=(80, 95)):
        # Check and prepare data
        x = x.asfreq(freq=period)
        pred = self.predict_ar(Theta, sigma2, x, n_ahead=h)

        # Form levels if not as iterable given
        level = [level] if isinstance(level, int) else level

        lower = np.empty((h, len(level)))
        upper = np.empty((h, len(level)))
        for i, conf_level in enumerate(level):
            qq = ppf(0.5 * (1 + conf_level / 100))
            lower[:, i] = pred["pred"] - qq * pred["se"]
            upper[:, i] = pred["pred"] + qq * pred["se"]

        # Convert the results
        result = {
            "level": level,
            "mean": pred["pred"],
            "se": pred["se"],
            "lower": lower,
            "upper": upper,
            "fitted": pred["fitted"],
            "residuals": pred["residuals"],
        }

        return result


# entry point for PySpark ETL application
if __name__ == '__main__':
    Darima().darima()

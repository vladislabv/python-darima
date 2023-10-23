"""
darima.py
~~~~~~~~~~
"""

# External Packages
import time
import json

import pandas as pd
import numpy as np
from rpy2 import robjects
from py_spark.spark import start_spark

# Internal Packages
from py_handlers.converters import convert_to_r_time_series, rvector_to_list_of_tuples, convert_result_to_df, \
    convert_spark_2_pandas_ts
from py_handlers.utils import ppf, inv_box_cox, ar_to_ma


class Darima:
    """
    Darima is the Meta Class for modelling.

    """

    def __init__(self, column_name_value: str = "demand", column_name_time: str = "time"):
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
        self.num_of_partitions = self.config_darima['num_partitions']
        self.frequency = self.config_darima['data_time_freq']
        self.method = self.config_darima["method"]
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
        train_data = self.load_data_from_csv(spark, self.config_darima['train_datapath'])
        test_data = self.load_data_from_csv(spark, self.config_darima['test_datapath'])

        data_transformed = self.map_reduce(train_data).collect()
        df_coeffs = convert_result_to_df(data_transformed)

        if self.config_darima['method'] == 'dlsa':
            temp_sigma = (df_coeffs[df_coeffs['coef'] == 'sigma2']["value"].values[0])
            df_coeffs["value"] = (df_coeffs["value"] * (1 / temp_sigma))/test_data.count()
            df_coeffs[df_coeffs['coef'] == 'sigma2'].loc[:, "value"] = (1 / temp_sigma) * test_data.count()

        elif self.config_darima["method"] == "mean":
            df_coeffs["value"] = df_coeffs["value"] / test_data.count()

        # before doing preds convert pyspark df to pandas df
        test_pd_ts = convert_spark_2_pandas_ts(test_data, self.column_name_time)
        train_pd_ts = convert_spark_2_pandas_ts(train_data, self.column_name_time)

        preds = ForecastDarima().forecast_darima(
            Theta=np.array(df_coeffs[df_coeffs['coef'] != 'sigma2']["value"].values),
            sigma2=df_coeffs[df_coeffs['coef'] == 'sigma2']["value"].values[0],
            x=train_pd_ts,
            period=self.frequency,
            h=len(test_pd_ts),
            level=[80, 95],
        )
        
        eval_metrics = EvaluationDarima().model_evaluation(
            log=log,
            train=train_pd_ts,
            test=test_pd_ts,
            period=self.frequency,
            pred=preds["mean"],
            lower=preds["lower"],
            upper=preds["upper"],
            level=[80, 95],
        )
        score = eval_metrics.mean(axis=0)
        print(score)
        # log the success and terminate Spark application
        log.warn('Darima is finished')
        spark.stop()
        return None

    def load_data_from_csv(self, spark, filename):
        """
        Load data from CSV file format.

        :param spark: Spark session object.
        :param filename: CSV filename
        :return: Spark DataFrame.
        """
        df = (
            spark
            .read
            .csv(filename, header=True, inferSchema=True)
        )

        return df

    def map_reduce(self, df):
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
            .mapPartitions(lambda x: MapDarima().map_arima(x))
            .flatMap(lambda x: x)
        )

        if self.config_darima['method'] == "dlsa":
            result = ReduceDarima().reduce_dlsa(converted_ts_rdd)
        elif self.config_darima['method'] == "mean":
            result = ReduceDarima().reduce_mean(converted_ts_rdd)
        else:
            result = None
        return result
    

class MapDarima(Darima):
    """
    Is the main class for the Map step. This class inherit Darima class.
    Mapping every partition and calculating the coefficients. Uses R intergrated functions
    like auto.arima and ts. Also preparing a partition for calculations and convertering
    given partition into a timeseries. map_arima is the main-method within this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        method = self.config_darima['method']
        arima_model_coefficients = r_auto_arima(ts, method)
        return rvector_to_list_of_tuples(arima_model_coefficients)


class ReduceDarima(Darima):
    """
    Is the main class for the Reduce step. This class inherit Darima class.
    Got two main-methods, reduce_dlsa (Distributed Least Squares Approximation) and reduce_mean.
    Will return processed coefficients.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce_dlsa(self, converted_ts_rdd):
        """
        Calculating for a RDD partition the sum for all key-value pairs.
        Is created for getting the sum of all the coefficients.

        :param converted_ts_rdd: Should be all the coefficients
        :type converted_ts_rdd: RDD Format
        :return: Sum over Keys
        :rtype: RDD Format
        """

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
        return sums_over_keys

    def reduce_mean(self, converted_ts_rdd):
        """
        Calculating mean of the ARIMA coefficients across all partitions

        :param converted_ts_rdd: Should be all the coefficients
        :type converted_ts_rdd: RDD Format
        :return: Result (reduced) list of tuples [(coef_1, value), (coef_2, value), ...]
        :rtype: RDD Format
        """
        # holds initial values for sum and count
        initial_value = (0, 0)

        calc_within_parts = lambda a, b: (a[0] + b, a[1] + 1)

        calc_cross_parts = lambda a, b: (a[0] + b[0], a[1] + b[1])
        mean_coeffs = converted_ts_rdd.aggregateByKey(initial_value, calc_within_parts, calc_cross_parts)

        # Finally, calculate the average for each KEY, and collect results.
        result = mean_coeffs.mapValues(lambda v: v[0] / v[1])

        return result

    
class ForecastDarima(Darima):
    """
    Is the main forecasting class. This class inherit Darima class. Is calculating the predictions of the
    coefficients of Darima. Is the main class for forecasting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_ar(self, Theta, sigma2, x, n_ahead=1, se_fit=True):
        """
        Predicts the values by applying autoregressive model.
        Optionally calculated standard errors for the made predictions over the given horizon.

        :param Theta: Array of the ARIMA-coefficients (except sigma)
        :param sigma2: Sigma value
        :param x: Train data to make the predictions on
        :param n_ahead Forecasting horizon
        :param se_fit: Whether standard errors should be included in the output
        :return: Dictionary including residuals, fitted values and predictions
        :rtype: dict
        """
        # Check arguments
        if n_ahead < 1:
            raise ValueError("'n_ahead' must be at least 1")
        if x is None:
            raise ValueError("Argument 'x' is missing")
        if not isinstance(x, pd.Series):
            raise ValueError("'x' must be a pandas time series")

        h = n_ahead
        n = len(x)
        tspx = x.index
        dt = (tspx[1] - tspx[0]).total_seconds()
        coef = Theta
        p = len(coef) - 2  # AR order

        X = np.column_stack([np.ones(n), np.arange(1, n + 1)] + [x.shift(i).to_numpy() for i in range(1, p + 1)])
        print(coef)
        # Fitted value
        fits = np.dot(X, coef).ravel()
        fits = pd.Series(fits, index=tspx)

        # Residuals
        res = x - fits

        # Forecasts
        y = np.append(x, np.zeros(h))
        for i in range(h):
            y[n + i] = np.sum(coef * np.concatenate(([1, n + i], y[n + i - np.arange(1, p + 1)])))

        pred = y[n:]
        pred = pd.Series(pred, index=[x.index[-1] + pd.Timedelta((i + 1) * dt, unit='s') for i in range(h)])

        result = {
            "fitted": fits,
            "residuals": res,
            "pred": pred
        }

        # Standard errors
        if se_fit:
            psi = ar_to_ma(coef[-p:], h - 1) ** 2
            vars = np.cumsum(np.concatenate((np.array([1]), psi)))
            se = np.sqrt(sigma2 * vars)[:h]
            se = pd.Series(se, index=[tspx[-1] + pd.Timedelta((i + 1) * dt, unit='s') for i in range(h)])
            result["se"] = se

        return result
    
    def forecast_darima(self, Theta, sigma2, x, period=24, h=1, level=[80, 95]):
        """
        Forecasts values over the given horizon in the future. The predictions are formed as pandas (Time-) series.
        The results of forecast are additionaly dumped in form of the json document for further analysis.

        :param Theta: Array of the ARIMA-coefficients (except sigma)
        :param sigma2: Sigma value
        :param x: Train data to make the predictions on
        :param period: Frequency of observations
        :param h: Forecasting horizon
        :param level: Confidention levels used to calculated confidential intervals for each predicted value
        :return: Dictionary with fitted, predictions, lower/upper bounds, and residuals
        :rtype: dict
        """
        # Check and prepare data
        pred = self.predict_ar(Theta=Theta, sigma2=sigma2, x=x, n_ahead=h)

        # Check and prepare levels
        level = np.array(level) if isinstance(level, list) else level
        level = np.array([level]) if isinstance(level, int) or isinstance(level, float) else level

        if not isinstance(level, np.ndarray):
            raise ValueError("Confidence limit must be a number or a list of numbers")

        if (min(level) > 0 and max(level) < 1):
            level < - 100 * level
        elif (min(level) < 0 or max(level) > 99.99):
            raise ValueError("Confidence limit out of range")

        lower = np.empty((h, len(level)))
        upper = np.empty((h, len(level)))
        for i, conf_level in enumerate(level):
            qq = ppf(0.5 * (1 + conf_level / 100))
            lower[:, i] = pred["pred"] - qq * pred["se"]
            upper[:, i] = pred["pred"] + qq * pred["se"]

        # Convert the results
        result_to_dump = {
            "level": level.tolist(),
            "mean": pred["pred"].values.tolist(),
            "se": pred["se"].tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "fitted": pred["fitted"].values.tolist(),
            "residuals": pred["residuals"].values.tolist(),
        }

        result = {
            "level": level,
            "mean": pred["pred"],
            "se": pred["se"],
            "lower": lower,
            "upper": upper,
            "fitted": pred["fitted"],
            "residuals": pred["residuals"],
        }

        with open("forecast.json", "w") as outfile:
            json.dump(result_to_dump, outfile)
            print("INFORMATION: Forecasts were DUMPED")

        return result

class EvaluationDarima(Darima):
    """
    Is the main forecasting class. This class inherit Darima class. Is calculating the result
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_evaluation(self, log, train, test, period, pred, lower, upper, level=[80, 95]):
        """
        Calculates the evaluation metrics between the predictions and actiual test data.
        Use additionally test data to calculate scaling coefficient for evaluation metrics.

        :param log: PySpark Logger
        :param train: Train data, used to train ARIMA-Models
        :param test: Test data
        :param period: Frequency of observations
        :param pred: Predicted values
        :param lower: Lower bounds for predicted values
        :param upper: Upper bounds for predicted values
        :param level: Values used for alpha calculation, therefore MSIS value one per level
        :type level: List, np.ndarray, int or float
        :return: Out DF with named evaluation metrics
        :rtype: pd.DataFrame
        """
        # Check and prepare levels
        level = np.array(level) if isinstance(level, list) else level
        level = np.array([level]) if isinstance(level, int) or isinstance(level, float) else level

        if not isinstance(level, np.ndarray):
            raise ValueError("Confidence limit must be a number or a list of numbers")

        time_index = test.index.values
        # Calculate MASE
        scaling = np.mean(np.abs(np.diff(np.array(train), period)))
        mase = np.abs(test.reset_index(drop=True) - pred.reset_index(drop=True)) / scaling
        mase.name = "mase"

        import matplotlib.pyplot as plt
        plt.plot(test.reset_index(drop=True).index, test.reset_index(drop=True), color="blue")
        plt.plot(pred.reset_index(drop=True).index, pred.reset_index(drop=True), color="red")
        plt.show()

        # Calculate sMAPE
        smape = np.abs(test.reset_index(drop=True) - pred.reset_index(drop=True)) / (
                    (np.abs(test.reset_index(drop=True)) + np.abs(pred.reset_index(drop=True))) / 2)
        smape.name = "smape"

        # Calculate MSIS
        alpha = (100 - level) / 100

        msis = pd.DataFrame()
        msis_names = ["msis_" + str(l) for l in level]
        for lower_col, upper_col, alpha, new_name in zip(lower.T, upper.T, alpha.T, msis_names):
            msis_col = (
                               (upper_col - lower_col) +
                               (2 / alpha) * (lower_col - test) * (lower_col > test) +
                               (2 / alpha) * (test - upper_col) * (upper_col < test)
                       ) / scaling
            msis_col.name = new_name
            msis = pd.concat([msis, msis_col], axis=1)

        # Out
        # --------------------------------------
        out_df = pd.concat([mase, smape, msis.reset_index(drop=True)], axis=1)
        out_df.index = time_index

        if any(out_df.isna()):
            log.warn("NAs appear in the final output")

        return out_df


# entry point for PySpark ETL application
if __name__ == '__main__':
    Darima().darima()

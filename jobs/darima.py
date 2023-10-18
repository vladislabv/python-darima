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
from rpy2 import robjects
from py_spark.spark import start_spark
import time
import json

# Internal Packages
from py_handlers.converters import convert_to_r_time_series, rvector_to_list_of_tuples, convert_result_to_df

class Darima:

    def __init__(self):
        with open("../configs/darima_config.json", 'r') as config_file:
            self.config_darima = json.load(config_file)


    def darima(self):
        """Main ETL script definition.

        :return: None
        """

        # start Spark application and get Spark session, logger and config
        spark, log, config = start_spark(
            app_name='DarimaModel',
            files=[])

        # log that main ETL job is starting
        log.warn('Darima job is up-and-running')

        # execute ETL (Darima) pipeline with Map and Reduce steps
        data = self.extract_data(spark)
        data_transformed = self.mapreduce_transform_data(
            data,
            num_partitions=self.config_darima['num_partitions'],
            frequency=self.config_darima['data_time_freq']
        ).collect()
        data_transformed = convert_result_to_df(data_transformed)
        print(data_transformed)


        #load_data(data_transformed)

        # log the success and terminate Spark application
        log.warn('Darima is finished')
        time.sleep(1000)
        spark.stop()
        return None


    def extract_data(self, spark):
        """Load data from CSV file format.

        :param spark: Spark session object.
        :return: Spark DataFrame.
        """
        df = (
            spark
            .read
            .csv("../data/CT_test.csv", header=True, inferSchema=True)
        )

        return df


    def mapreduce_transform_data(self, df, num_partitions, frequency):
        """Transform original dataset.

        :param df: Input DataFrame.
        :param steps_per_floor_: The number of steps per-floor at 43 Tanner
            Street.
        :return: Transformed DataFrame.
        """
        parts = (
            df
           .repartition(num_partitions)
           .rdd
        )

        converted_ts_rdd = (
            parts
            .mapPartitions(lambda x: self.map_arima(x, frequency))
            .flatMap(lambda x: x)
        )
        # holds initial values for sum and count
        aTuple = (0,0)
        # First lambda expression for Within-Partition Reduction Step::
        # a: is a TUPLE that holds: (runningSum, runningCount).
        # b: is a SCALAR that holds the next Value
        calc_within_parts = lambda a, b: (a[0] + b, a[1] + 1)
        # Second lambda expression for Cross-Partition Reduction Step::
        # a: is a TUPLE that holds: (runningSum, runningCount).
        # b: is a TUPLE that holds: (nextPartitionsSum, nextPartitionsCount).
        calc_cross_parts = lambda a, b: (a[0] + b[0], a[1] + b[1])
        mean_coeffs = converted_ts_rdd.aggregateByKey(aTuple, calc_within_parts, calc_cross_parts)

        # Finally, calculate the average for each KEY, and collect results.
        result = mean_coeffs.mapValues(lambda v: v[0]/v[1])
        return result


    def process_data(self, df):
        pass


    def load_data(self, df):
        """Collect data locally and write to CSV.

        :param df: DataFrame to print.
        :return: None
        """
        (df
         .coalesce(1)
         .write
         .csv('loaded_data', mode='overwrite', header=True))
        return None


    # Map functions
    def map_arima(self, iterator, frequency=24):
        rows = list(iterator)
        demand_values = [row["demand"] for row in rows]
        time_values = [str(row["time"]) for row in rows]
        ts = convert_to_r_time_series(demand_values, time_values, frequency)
        result = self.auto_arima(ts)
        yield result



    def auto_arima(self, ts):
        """
        Creating via auto_arima

        :param ts:
        :type ts:
        :return:
        :rtype:
        """
        robjects.r.source("../R/auto_arima.R")
        r_auto_arima = robjects.r["auto_arima"]
        arima_model_coefficients = r_auto_arima(ts)
        return rvector_to_list_of_tuples(arima_model_coefficients)

# entry point for PySpark ETL application
if __name__ == '__main__':
    Darima().darima()

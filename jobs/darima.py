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

from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit
from rpy2 import robjects

from dependencies.spark import start_spark
from py_handlers.converters import convert_to_r_time_series, rvector_to_list_of_tuples
import time
import pprint

def main():
    """Main ETL script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='spark.py',
        files=['configs/etl_config.json'])

    # log that main ETL job is starting
    log.warn('Darima job is up-and-running')

    # execute ETL (Darima) pipeline with Map and Reduce steps
    data = extract_data(spark)
    data_transformed = mapreduce_transform_data(
        data, 
        num_partitions=config['num_partitions'],
        frequency=config['data_time_freq']
    ).collect()
    print(data_transformed)
    
    #load_data(data_transformed)

    # log the success and terminate Spark application
    log.warn('Darima is finished')
    time.sleep(1000)
    spark.stop()
    return None


def extract_data(spark):
    """Load data from CSV file format.

    :param spark: Spark session object.
    :return: Spark DataFrame.
    """
    df = (
        spark
        .read
        .csv("data/CT_test.csv", header=True, inferSchema=True)
    )

    return df


def mapreduce_transform_data(df, num_partitions, frequency):
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
        .mapPartitions(lambda x: map_arima(x, frequency))
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


def process_data(df):
    pass


def load_data(df):
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
def map_arima(iterator, frequency=1):
    rows = list(iterator)
    demand_values = [row["demand"] for row in rows]
    time_values = [str(row["time"]) for row in rows]
    ts = convert_to_r_time_series(demand_values, time_values, frequency)
    trained_model = auto_arima(ts)
    yield trained_model


def auto_arima(ts):
    # import needed R objects
    robjects.r.source("R/auto_arima.R")
    r_auto_arima = robjects.r["auto_arima"]
    # r_forecast_arima = robjects.r["forecast_arima"]
    arima_model_coefficients = r_auto_arima(ts)
    # forecasted_values = r_forecast_arima(arima_model, ts)
    return rvector_to_list_of_tuples(arima_model_coefficients)


# Reduce functions


def create_test_data(spark, config):
    """Create test data.

    This function creates both both pre- and post- transformation data
    saved as Parquet files in tests/test_data. This will be used for
    unit tests as well as to load as part of the example ETL job.
    :return: None
    """
    # create example data from scratch
    local_records = [
        Row(id=1, first_name='Dan', second_name='Germain', floor=1),
        Row(id=2, first_name='Dan', second_name='Sommerville', floor=1),
        Row(id=3, first_name='Alex', second_name='Ioannides', floor=2),
        Row(id=4, first_name='Ken', second_name='Lai', floor=2),
        Row(id=5, first_name='Stu', second_name='White', floor=3),
        Row(id=6, first_name='Mark', second_name='Sweeting', floor=3),
        Row(id=7, first_name='Phil', second_name='Bird', floor=4),
        Row(id=8, first_name='Kim', second_name='Suter', floor=4)
    ]

    df = spark.createDataFrame(local_records)

    # write to Parquet file format
    (df
     .coalesce(1)
     .write
     .parquet('tests/test_data/employees', mode='overwrite'))

    # create transformed version of data
    df_tf = mapreduce_transform_data(df, config['steps_per_floor'])

    # write transformed version of data to Parquet
    (df_tf
     .coalesce(1)
     .write
     .parquet('tests/test_data/employees_report', mode='overwrite'))

    return None


# entry point for PySpark ETL application
if __name__ == '__main__':
    main()

import pandas as pd
import rpy2.robjects as robjects

def convert_to_r_time_series(data: list,
                             data_time: list,
                             frequency: int = 12,
                             is_datetime: bool = True,
                             format: str = "%Y-%m-%d %H:%M:%S"):


    """
    Converts data to a FloatVector in R, creating a time series object.
    data.index need to be frequently (Timestamps / frequence)
    This need to be passed into auto.arima()

    :param data: Values / f(x)
    :type data: pd.Series
    :param data_time: Time-Values / x
    :type data_time: pd.Series
    :param frequency: 4: Quarterly, 12 Monthly, 24 Hourly, 60 Minutely, 3600 Secondly
    :type frequency: int
    :param is_datetime: is_date checking, if its correct
    :type is_datetime: Boolean
    :param format: dateformat
    :type format: str
    :return: TimeSeries in R
    :rtype: robjects.r["ts"] <- This type
    """
    # Convert the pandas Series to an R FloatVector
    converted_object = robjects.FloatVector(data)

    # Convert the pandas Series index to R Date format
    if is_datetime:
        r_date_vector = robjects.StrVector(data_time)
        frequence_index = robjects.r['as.POSIXct'](r_date_vector, format=format)

    else:
        print("INFORMATION: data.index should be a datetime-format")
        frequence_index = robjects.IntVector(data_time)

    # Create a time series object with the converted data and time index
    ts_data = robjects.r["ts"](converted_object, start=min(frequence_index), frequency=frequency)
    is_ts = robjects.r["is.ts"](ts_data)

    print(f"INFORMATION: Is a TimeSeries: {is_ts}")

    return converted_object


def rvector_to_list_of_tuples(r_vector):
    # Convert R vector to a list

    python_list = robjects.conversion.rpy2py(r_vector)
    
    # Get names of the R vector
    names = robjects.r.names(r_vector)
    
    # Create a list of tuples with name-value pairs
    result = [(names[i], python_list[i]) for i in range(len(python_list))]

    return result


def convert_result_to_df(result):
    """
    Will convert the results into structured DataFrames.

    from: [(coef_1, value), (coef_2, value2)...]
    to: Pandas DataFrames

    Will also split into 3 DataFrames.

    - One for the ar coefficients
    - One for the sigma coefficients
    - One for the beta coefficients

    :param result: results coefficients
    :type result: list of tuples
    :return: df_ar, df_sigma, df_beta
    :rtype: pd.DataFrame, pd.DataFrame, pd.DataFrame
    """
    df = pd.DataFrame(result, columns=["coef", "value"])
    df_ar = df.loc[df["coef"].str.contains("ar")]
    df_ar.loc[:, 'coef'] = "ar_" + df_ar['coef'].str.split("ar").str[-1].astype(int).apply(lambda x: f'{x:08d}')
    df_ar = df_ar.sort_values(by="coef").reset_index(drop=True)
    # combine sigma and betas coefficients
    df_sigma = df.loc[df["coef"].str.contains("sigma")].reset_index(drop=True)
    df_beta = df.loc[df["coef"].str.contains("beta")].reset_index(drop=True)
    # combine all dataframes
    result = pd.concat([df_ar, df_sigma, df_beta], axis=0).reset_index(drop=True)

    return result

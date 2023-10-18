import pandas as pd
import rpy2.robjects as robjects

def convert_to_r_time_series(data: list,
                             data_time: list,
                             frequency: int = 12,
                             is_datetime: bool = True,
                             format: str = "%Y-%m-%d %H:%M:%S") -> robjects.r["ts"]:
    """
    data: pd.Series -> Values / f(x)
    data_time: pd.Series -> Time-Values / x
    frequency: int -> 4: Quarterly, 12 Monthly, 24 Hourly, 60 Minutely, 3600 Secondly
    is_date: Boolean
    format: String

    return: TimeSeries in R

    Converts data to a FloatVector in R, creating a time series object.
    data.index need to be frequently (Timestamps / frequence)
    This need to be passed into auto.arima()
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
    df = pd.DataFrame(result, columns=["coef", "value"])
    df_ar = df.loc[df["coef"].str.contains("ar")]
    df_ar.loc[:, 'coef'] = "ar_" + df_ar['coef'].str.split("ar").str[-1].astype(int).apply(lambda x: f'{x:08d}')
    df_ar = df_ar.sort_values(by="coef").reset_index(drop=True)
    df_sigma = df.loc[df["coef"].str.contains("sigma")].reset_index(drop=True)
    df_beta = df.loc[df["coef"].str.contains("beta")].reset_index(drop=True)
    return df_ar, df_sigma, df_beta


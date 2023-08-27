import rpy2.robjects as robjects
import pandas as pd

def convert_to_r_time_series(data: pd.Series,
                             frequency: int = 12,
                             is_datetime: bool = True,
                             format: str = "%Y-%m-%d %H:%M:%S") -> robjects.r["ts"]:
    """
    data: pd.Series
    frequency: int -> 4: Quarterly, 12 Monthly, 24 Hourly, 60 Minutely, 3600 Secondly
    is_date: Boolean
    format: String

    return: TimeSeries in R

    Converts data to a FloatVector in R, creating a time series object.
    data.index need to be frequently (Timestamps / frequence)
    This need to be passed into auto.arima()
    """
    # Convert the pandas Series to an R FloatVector
    converted_object = robjects.FloatVector(data.values.tolist())

    # Convert the pandas Series index to R Date format
    if is_datetime:
        r_date_vector = robjects.StrVector(data.index.tolist())
        frequence_index = robjects.r['as.POSIXct'](r_date_vector, format=format)

    else:
        print("INFORMATION: data.index should be a datetime-format")
        frequence_index = robjects.IntVector(data.index.tolist())

    # Create a time series object with the converted data and time index
    ts_data = robjects.r["ts"](converted_object, start=min(frequence_index), frequency=frequency)
    is_ts = robjects.r["is.ts"](ts_data)

    print(f"INFORMATION: Is a TimeSeries: {is_ts}")

    return ts_data


# # Example DataFrame
# data = pd.Series([100, 120, 130, 160,
#                   100, 120, 130, 160,
#                   100, 120, 130, 160,
#                   100, 120, 130, 160,
#                   100, 120, 130, 160,
#                   100, 120, 130, 160,
#                 160
#                   ],
#                  index=["2017-01-01 00:00:00", "2017-01-01 01:00:00", "2017-01-01 02:00:00", "2017-01-01 03:00:00",
#                         "2017-01-01 04:00:00", "2017-01-01 05:00:00", "2017-01-01 06:00:00", "2017-01-01 07:00:00",
#                         "2017-01-01 08:00:00", "2017-01-01 09:00:00", "2017-01-01 10:00:00", "2017-01-01 11:00:00",
#                         "2017-01-01 12:00:00", "2017-01-01 13:00:00", "2017-01-01 14:00:00", "2017-01-01 15:00:00",
#                         "2017-01-01 16:00:00", "2017-01-01 17:00:00", "2017-01-01 18:00:00", "2017-01-01 19:00:00",
#                         "2017-01-01 20:00:00", "2017-01-01 21:00:00", "2017-01-01 22:00:00", "2017-01-01 23:00:00",
#                         "2017-01-02 00:00:00"
#                         ])
#
# # Convert DataFrame to R time series
# frequency = 24  # Assuming monthly data
# r_time_series = convert_to_r_time_series(data, frequency, is_datetime=True)
# print(r_time_series)

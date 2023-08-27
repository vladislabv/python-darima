import rpy2.robjects as robjects
import pandas as pd

def convert_to_r_time_series(data: pd.Series,
                             frequency: int = 12,
                             is_datetime: bool = True,
                             format: str = "%Y-%m-%d") -> robjects.r["ts"]:
    """
    data: pd.Series
    frequency: int
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
        frequence_index = robjects.r["as.Date"](r_date_vector, format=format)

    else:
        print("INFORMATION: data.index should be a datetime-format")
        frequence_index = robjects.IntVector(data.index.tolist())

    # Create a time series object with the converted data and time index
    ts_data = robjects.r["ts"](converted_object, start=min(frequence_index), frequency=frequency)
    is_ts = robjects.r["is.ts"](ts_data)

    print(f"INFORMATION: Is a TimeSeries: {is_ts}")

    return ts_data


# Example DataFrame
data = pd.Series([100, 120, 130, 160], index=["1","2","3","5"])

# Convert DataFrame to R time series
frequency = 10  # Assuming monthly data
r_time_series = convert_to_r_time_series(data, frequency, is_datetime=True, format="%H")

# Print the R time series
print(r_time_series)

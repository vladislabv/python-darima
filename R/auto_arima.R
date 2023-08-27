auto_arima <- function(train_data, test_data){

    # train_data should be convert_to_r_time_series()
    # test_data should be convert_to_r_time_series()
    # Is a TimeSeries object
    # Will find via auto.arima best parameters for the arima model

    library(forecast)

    # Fit eines ARIMA-Modells
    arima_model <- auto.arima(train_data)

    # Prognose für die Testdaten
    forecast_values <- forecast(arima_model, h = length(test_data))

    return (forecast_values)

    }



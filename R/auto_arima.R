auto_arima <- function(train_data){

    # train_data should be convert_to_r_time_series()
    # test_data should be convert_to_r_time_series()
    # Is a TimeSeries object
    # Will find via auto.arima best parameters for the arima model

    library(forecast)

    # Fit eines ARIMA-Modells
    arima_model <- auto.arima(train_data)

    return (arima_model)
    }

forecast_arima <- function(arima_model, test_data){
    library(forecast)

     # Prognose für die Testdaten
    forecast_values <- forecast(arima_model, h = length(test_data))

    write.csv(forecast_values, file = "forecasted_values.csv", row.names = FALSE)
    return (forecast_values)

    }



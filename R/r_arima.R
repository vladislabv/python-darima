# pip install rpy2


# Laden der erforderlichen Bibliotheken
test <- function(){

    library(forecast)

    # Laden der Daten
    data("AirPassengers")

    # Aufteilen der Daten in Trainings- und Testdaten
    train_data <- window(AirPassengers, end=c(1958,12))
    test_data <- window(AirPassengers, start=c(1959,1))

    # Fit eines ARIMA-Modells
    arima_model <- auto.arima(train_data)

    # Prognose für die Testdaten
    forecast_values <- forecast(arima_model, h = length(test_data))

    # Ausgabe der Prognoseergebnisse
    print(forecast_values)
    }



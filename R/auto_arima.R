suppressPackageStartupMessages(require("forecast"))
suppressPackageStartupMessages(require("polynom"))

ar_coefficients <- function(ar = 0, d = 0L, ma = 0, 
                            sar = 0, D = 0L, sma = 0, 
                            mean = 0, drift = 0, 
                            m = 1L, tol = 500L) {
    mu <- mean
    dft <- drift

    # non-seasonal AR
    ar <- polynomial(c(1, -ar)) * polynomial(c(1, -1))^d

    # seasonal AR
    if (m > 1) {
        P <- length(sar)
        seasonal_poly <- numeric(m * P)
        seasonal_poly[m * seq(P)] <- sar
        sar <- polynomial(c(1, -seasonal_poly)) * polynomial(c(1, rep(0, m - 1), -1))^D
    }
    else {
        sar <- 1
    }

    # non-seasonal MA
    ma <- polynomial(c(1, ma))

    # seasonal MA
    if (m > 1) {
        Q <- length(sma)
        seasonal_poly <- numeric(m * Q)
        seasonal_poly[m * seq(Q)] <- sma
        sma <- polynomial(c(1, seasonal_poly))
    }
    else {
        sma <- 1
    }

    # pie
    n <- tol
    theta <- -c(coef(ma * sma))[-1]
    if (length(theta) == 0L) {
        theta <- 0
    }
    phi <- -c(coef(ar * sar)[-1], numeric(n))
    q <- length(theta)
    pie <- c(numeric(q), 1, numeric(n))
    for (j in seq(n)) {
        pie[j + q + 1L] <- -phi[j] + sum(theta * pie[(q:1L) + j])
    }
    pie <- pie[(0L:n) + q + 1L]
    pie <- head(pie, (tol+1)) 
    pie <- -pie[-1]

    c0 <- mu * (1 - sum(pie)) + dft * (t(seq_len(tol)) %*% pie)
    c1 <- dft * (1 - sum(pie))

    # y_t = c0 + c1 * t + pie_1 * y_{t-1} + ... + pie_tol * y_{t-tol} + epsilon_t
    coef <- `names<-` (
        c(c0, c1, pie), 
        c("beta0", "beta1", paste("ar", sep = "", seq_len(tol)))
    )

    return(coef)
}

auto_arima <- function(train_data){

    # train_data should be convert_to_r_time_series()
    # test_data should be convert_to_r_time_series()
    # Is a TimeSeries object
    # Will find via auto.arima best parameters for the arima model

    tol <- 2000

    # Fit eines ARIMA-Modells
    arima_model <- auto.arima(train_data)
    sigma2 <- c(arima_model$sigma2)

    d <- arima_model$arma[6]
    D <- arima_model$arma[7]
    m <- arima_model$arma[5]
    mu <- arima_model$coef[names(arima_model$coef) == "intercept"]
    if (length(mu)==0) {
        mu <- 0
    }
    dft <- arima_model$coef[names(arima_model$coef) == "drift"]
    if (length(dft)==0) {
        dft <- 0
    }
    phi <- arima_model$coef[substring(names(arima_model$coef), 1, 2) == "ar"]
    theta <- arima_model$coef[substring(names(arima_model$coef), 1, 2) == "ma"]
    Phi <- arima_model$coef[substring(names(arima_model$coef), 1, 3) == "sar"]
    Theta <- arima_model$coef[substring(names(arima_model$coef), 1, 3) == "sma"]

    ar.coef <- ar_coefficients(
        ar = phi, d = d, ma = theta, 
        sar = Phi, D = D, sma = Theta, 
        mean = mu, drift = dft, 
        m = m, tol = tol
    )
    # append sigma to the resulting vector
    ar.coef["sigma2"] <- sigma2
    # should be named vector
    return(ar.coef)
}

forecast_arima <- function(arima_model, test_data){

    # Prognose für die Testdaten
    forecast_values <- forecast(arima_model, h = length(test_data))

    write.csv(forecast_values, file = "forecasted_values.csv", row.names = FALSE)
    return (forecast_values)

}
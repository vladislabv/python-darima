import rpy2.robjects as robjects


# Load the R script containing your function
robjects.r.source("R/r_arima.R")

# Get the R function
my_r_function = robjects.r["test"]


my_r_function()


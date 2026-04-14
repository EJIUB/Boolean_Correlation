args <- commandArgs(trailingOnly = TRUE)

# Usage:
#   Rscript run_wlasso.R X.csv y.csv out.json [gamma] [maxsteps]
x_path   <- args[1]
y_path   <- args[2]
out_path <- args[3]
gamma    <- if (length(args) >= 4) as.numeric(args[4]) else 0.95
maxsteps <- if (length(args) >= 5) as.integer(args[5]) else 2000

suppressWarnings(
  suppressPackageStartupMessages({

    # ---- WLasso + dependency ----
    if (!requireNamespace("WLasso", quietly = TRUE)) {
      if (!requireNamespace("genlasso", quietly = TRUE)) {
        install.packages("genlasso", repos = "https://cloud.r-project.org")
      }
      install.packages(
        "https://cran.r-project.org/src/contrib/Archive/WLasso/WLasso_1.0.tar.gz",
        repos = NULL,
        type = "source"
      )
    }
    library(WLasso)

    # ---- jsonlite ----
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      install.packages("jsonlite", repos = "https://cloud.r-project.org")
    }
    library(jsonlite)

  })
)

X <- as.matrix(read.csv(x_path, header = FALSE))
Y <- as.numeric(read.csv(y_path, header = FALSE)[, 1])

Sigma_est <- Sigma_Estimation(X)
mod <- Whitening_Lasso(X = X, Y = Y, Sigma = Sigma_est$mat, gamma = gamma, maxsteps = maxsteps)

beta_min <- mod$beta.min
selected <- which(beta_min != 0)

out <- list(
  method = "WLasso",
  gamma = gamma,
  maxsteps = maxsteps,
  beta_hat = as.numeric(beta_min),
  selected_1based = as.integer(selected),
  mse = as.numeric(mod$mse),
  alpha_hat = as.numeric(Sigma_est$alpha),
  group_act_1based = as.integer(Sigma_est$group_act)
)

write_json(out, out_path, auto_unbox = TRUE)

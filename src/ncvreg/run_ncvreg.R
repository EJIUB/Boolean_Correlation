args <- commandArgs(trailingOnly=TRUE)
x_path <- args[1]
y_path <- args[2]
out_path <- args[3]

penalty <- if (length(args) >= 4) args[4] else "MCP"     # "MCP","SCAD","lasso"
nfolds  <- if (length(args) >= 5) as.integer(args[5]) else 5
seed    <- if (length(args) >= 6) as.integer(args[6]) else 1

# Grids as comma-separated strings
gamma_grid_str <- if (length(args) >= 7) args[7] else ""
alpha_grid_str <- if (length(args) >= 8) args[8] else ""

# If provided, skip CV and fit at this lambda (runs 1..9)
lambda_fixed <- if (length(args) >= 9) as.numeric(args[9]) else NA
gamma_fixed  <- if (length(args) >= 10) as.numeric(args[10]) else NA
alpha_fixed  <- if (length(args) >= 11) as.numeric(args[11]) else NA

suppressPackageStartupMessages({
  library(jsonlite)
  library(ncvreg)
})
    
    
suppressWarnings(
  suppressPackageStartupMessages({

    # ---- WLasso + dependency ----
    if (!requireNamespace("ncvreg", quietly = TRUE)) {
      install.packages("ncvreg")
    }
    library(ncvreg)

    # ---- jsonlite ----
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      install.packages("jsonlite", repos = "https://cloud.r-project.org")
    }
    library(jsonlite)

  })
)
    

X <- as.matrix(read.csv(x_path, header=FALSE))
y <- as.numeric(read.csv(y_path, header=FALSE)[,1])

set.seed(seed)

parse_grid <- function(s) {
  if (nchar(s) == 0) return(NULL)
  as.numeric(strsplit(s, ",")[[1]])
}

gamma_grid <- parse_grid(gamma_grid_str)
alpha_grid <- parse_grid(alpha_grid_str)

# Defaults if grids not provided
if (is.null(alpha_grid)) alpha_grid <- c(1.0, 0.9, 0.7, 0.5, 0.3, 0.1)

# For lasso, gamma irrelevant
if (penalty == "lasso") {
  gamma_grid <- c(NA_real_)
} else {
  if (is.null(gamma_grid)) {
    gamma_grid <- if (penalty == "SCAD") c(2.5, 3.7, 5, 10) else c(1.5, 3, 5, 10)
  }
}

# ---- Mode 1: fixed fit (runs 1..9) ----
if (!is.na(lambda_fixed)) {
  if (is.na(alpha_fixed)) alpha_fixed <- 1.0
  if (penalty != "lasso" && is.na(gamma_fixed)) gamma_fixed <- if (penalty == "SCAD") 3.7 else 3.0

  fit <- ncvreg(X, y, family="gaussian", penalty=penalty,
                alpha=alpha_fixed,
                gamma=if (penalty == "lasso") 3 else gamma_fixed,
                lambda=lambda_fixed)

  # b <- as.numeric(fit$beta)
  b <- as.numeric(fit$beta[, 1])
  selected <- which(b != 0)

  out <- list(
    penalty=penalty,
    alpha=as.numeric(alpha_fixed),
    # gamma=if (penalty=="lasso") NULL else as.numeric(gamma_fixed),
    gamma=if (penalty=="lasso") NA_real_ else as.numeric(gamma_fixed),
    lambda=as.numeric(lambda_fixed),
    beta_hat=b,
    selected_1based=as.integer(selected),
    cve_min=NULL
  )
  write_json(out, out_path, auto_unbox=TRUE)
  quit(save="no")
}

# ---- Mode 2: grid tuning (run 0) ----
best <- list(score=Inf, alpha=NA, gamma=NA, lambda=NA, beta=NULL, selected=NULL)

for (a in alpha_grid) {
  for (g in gamma_grid) {
    cvfit <- cv.ncvreg(
      X, y,
      family="gaussian",
      penalty=penalty,
      alpha=a,
      gamma=if (penalty == "lasso") 3 else g,
      nfolds=nfolds
    )

    score <- min(cvfit$cve)
    lam <- cvfit$lambda.min
    b <- as.numeric(coef(cvfit, lambda=lam))[-1]  # drop intercept
    sel <- which(b != 0)

    if (score < best$score) {
      best$score <- score
      best$alpha <- a
      best$gamma <- if (penalty=="lasso") NA_real_ else g
      best$lambda <- lam
      best$beta <- b
      best$selected <- sel
    }
  }
}

out <- list(
  penalty=penalty,
  alpha=as.numeric(best$alpha),
  # gamma=if (penalty=="lasso") NULL else as.numeric(best$gamma),
  gamma=if (penalty=="lasso") NA_real_ else as.numeric(best$gamma),
  lambda=as.numeric(best$lambda),
  beta_hat=as.numeric(best$beta),
  selected_1based=as.integer(best$selected),
  cve_min=as.numeric(best$score)
)

write_json(out, out_path, auto_unbox=TRUE)
args <- commandArgs(trailingOnly=TRUE)
x_path <- args[1]
y_path <- args[2]
out_path <- args[3]
num_select <- if (length(args) >= 4) as.integer(args[4]) else NA
family <- if (length(args) >= 5) args[5] else "gaussian"

    
suppressWarnings(
  suppressPackageStartupMessages({
      
    library(devtools)

    # ---- WLasso + dependency ----
    if (!requireNamespace("screening", quietly = TRUE)) {
        install_github('wwrechard/screening')
    }
    library(screening)

    # ---- jsonlite ----
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      install.packages("jsonlite", repos = "https://cloud.r-project.org")
    }
    library(jsonlite)

  })
)

X <- as.matrix(read.csv(x_path, header=FALSE))
y <- as.numeric(read.csv(y_path, header=FALSE)[,1])

if (is.na(num_select)) {
  num_select <- max(1L, floor(nrow(X) / 2))
}

res <- screening(x=X, y=y, method="holp", num.select=num_select, family=family)

# Most versions return res$screen (the vignette you showed earlier used output$screen)
sel <- NULL
for (nm in c("screen", "selected", "index", "idx", "keep")) {
  if (!is.null(res[[nm]])) { sel <- res[[nm]]; break }
}
if (is.null(sel) && is.numeric(res)) sel <- res

out <- list(
  method = "HOLP",
  family = family,
  num_select = as.integer(num_select),
  selected_1based = if (is.null(sel)) integer(0) else as.integer(sel)
)

write_json(out, out_path, auto_unbox=TRUE)
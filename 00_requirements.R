################################################################################
# Setup Script: Install and Load Required Dependencies for Climate Mortality
################################################################################

# Define all required CRAN packages
cran_packages <- c(
  "dplyr", "tidyr", "data.table", "dtplyr", "arrow", "stringr",
  "openxlsx", "dlnm", "doParallel", "doSNOW", "mixmeta", 
  "collapse", "ggplot2", "patchwork", "viridis", "ggnewscale", 
  "scico", "giscoR", "scales", "flextable", "ggdist", "ggtext", 
  "lemon"
)

# GitHub-only packages
github_packages <- list(
  PHEindicatormethods = "PublicHealthEngland/PHEindicatormethods",
  zen4R = "eblondel/zen4R"
)

# Install remotes if needed
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Install missing CRAN packages
missing_cran <- cran_packages[!cran_packages %in% installed.packages()]
if (length(missing_cran) > 0) {
  install.packages(missing_cran)
}

# Install missing GitHub packages
for (pkg in names(github_packages)) {
  if (!pkg %in% installed.packages()) {
    remotes::install_github(github_packages[[pkg]])
  }
}

# Load all packages
all_packages <- c(cran_packages, names(github_packages))
invisible(lapply(all_packages, library, character.only = TRUE))

# Check presence of function scripts
required_scripts <- c("functions/isimip3.R", "functions/impact.R")
missing_scripts <- required_scripts[!file.exists(required_scripts)]
if (length(missing_scripts) > 0) {
  stop("Missing required function scripts: ", paste(missing_scripts, collapse = ", "))
} else {
  source("functions/isimip3.R")
  source("functions/impact.R")
}

# Create data directory if needed and download from Zenodo
if (!dir.exists("data")) {
  dir.create("data")
  
  # Increase timeout and download
  zen4R::download_zenodo("10.5281/zenodo.14004321", path = "data", timeout = 10000, files = "data.zip")
  
  # Unzip and cleanup
  unzip("data/data.zip", exdir = "data")
  unlink("data/data.zip")
}

message("✅ All packages installed, data downloaded, and scripts loaded successfully.")

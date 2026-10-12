repos <- c(
  INLA = "https://inla.r-inla-download.org/R/stable",
  CRAN = "https://cloud.r-project.org"
)
install.packages("INLA", repos = repos, dep = TRUE)

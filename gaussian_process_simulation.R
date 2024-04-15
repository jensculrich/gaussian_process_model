# run through the example written by M. Betancourt found here:
# https://betanalpha.github.io/assets/case_studies/gaussian_processes.html

# set up the R environment

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

util <- new.env()

par(family="CMU Serif", las=1, bty="l", cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 5))

#---------------------------------------------
# 2.1 Simulating From A Gaussian Process 

# define the covariate grid 
N = 551
x <- 22 * (0:(N - 1)) / (N - 1) - 11

# define the parameters that specify our gaussian process
alpha_true <- 3
rho_true <- 5.5

# package everything together
simu_data <- list(alpha=alpha_true, rho=rho_true, N=N, x=x)

# in order to sample from the Gaussian process projected onto our covariate grid  
# we just need to construct the Gram matrix and then sample from the corresponding 
# multivariate normal random number generator.
writeLines(readLines("stan_programs/simu.stan"))

simu_fit <- stan(file='stan_programs/simu.stan', data=simu_data,
                 warmup=0, iter=4000, chains=1, seed=494838,
                 algorithm="Fixed_param", refresh=4000)

source('gp_utility.R', local=util)
# overlay some of the functions in a spaghetti plot,
util$plot_gp_prior_realizations(simu_fit, x, "Realizations")
# plot marginal quantiles to summarize the aggregate behavior
util$plot_gp_prior_quantiles(simu_fit, x, "Marginal Quantiles")

#---------------------------------------------
# 2.2 Simulating From A Gaussian Process Model

# Once we've specified a true measurement variability we can simulate 
# observations by sequentially sampling the function values along the
# covariate grid and then the observations at each of those points.

sigma_true <- 2
simu_data$sigma <- sigma_true

set.seed(2595)
simu_fit <- stan(file='stan_programs/simu_normal.stan', data=simu_data,
                 warmup=0, iter=4000, chains=1, seed=494838,
                 algorithm="Fixed_param", refresh=4000)

# Let's grab the first simulation for closer inspection.
f <- extract(simu_fit)$f[1,]
y <- extract(simu_fit)$y[1,]

c_mid_teal="#487575"
  
plot(x, f, type="l", lwd=2, xlab="x", ylab="y",
     xlim=c(-11, 11), ylim=c(-10, 10))
points(x, y, col="white", pch=16, cex=0.6)
points(x, y, col=c_mid_teal, pch=16, cex=0.4)

# define a measurement by the observations at just eleven evenly spaced points along the covariate grid.
observed_idx <- c(50 * (0:10) + 26)
N_obs = length(observed_idx)
x_obs <- x[observed_idx]
y_obs <- y[observed_idx]

N_predict <- N
x_predict <- x
y_predict <- y

plot(x, f, type="l", lwd=2, xlab="x", ylab="y",
     xlim=c(-11, 11), ylim=c(-10, 10))
points(x_predict, y_predict, col="white", pch=16, cex=0.6)
points(x_predict, y_predict, col=c_mid_teal, pch=16, cex=0.4)
points(x_obs, y_obs, col="white", pch=16, cex=1.2)
points(x_obs, y_obs, col="black", pch=16, cex=0.8)

# In order to use this simulation in the next section we'll put everything in a safe place.
stan_rdump(c("N_obs", "x_obs", "y_obs",
             "N_predict", "x_predict", "y_predict",
             "observed_idx"), file="output/normal.data.R")

stan_rdump(c("f", "x"), file="output/gp.truth.R")

#---------------------------------------------
# 2.3 Fitting A General Gaussian Process Posterior

truth <- read_rdump("output/gp.truth.R")

data <- read_rdump("output/normal.data.R")
data$alpha <- alpha_true
data$rho <- rho_true
data$sigma <- sigma_true

normal_fit <- stan(file='stan_programs/fit_normal.stan', data=data,
                   seed=5838299, refresh=1000)

# To check the fit let's load the stan_utility.R script for its diagnostic functions.
source('stan_utility.R', local=util)

writeLines(capture.output(util$check_n_eff(normal_fit))[1:5])
writeLines(capture.output(util$check_rhat(normal_fit))[1:5])
util$check_div(normal_fit)
util$check_treedepth(normal_fit)
util$check_energy(normal_fit)

# Unfortunately the diagnostics clearly indicate that Stan is not able to 
# accurately quantify the posterior for the projected function values. 
# We may be able to moderate the computational problems, however, by
# employing a non-centered parameterization of the function values.

normal_fit <- stan(file='stan_programs/fit_normal_ncp.stan', data=data,
                   seed=5838298, refresh=1000)

util$check_all_diagnostics(normal_fit)

util$plot_gp_post_realizations(normal_fit, data, truth,
                               "Posterior Realizations")

util$plot_gp_post_quantiles(normal_fit, data, truth,
                            "Posterior Marginal Quantiles")

util$plot_gp_post_pred_quantiles(normal_fit, data, truth,
                                 "Posterior Predictive Marginal Quantiles")

rm(list=ls())
library(rstan)
setwd("C:/Users/Caleb/Desktop/479/Project/stan_files")

###LOAD_DATA####

# median data
load("stan_dat/stan_dat.Rds")

# sd matrix
load("stan_dat/sd_mat.Rds")

# type coefs
load("stan_dat/type_coefs.Rds")
type.coefs <- as.numeric(type.coefs)

# zone coefs
load("stan_dat/zone_coefs.Rds")
zone.coefs <- as.numeric(zone.coefs)

# year coefs
load("stan_dat/year_coefs.Rds")
year.coefs <- as.numeric(year.coefs)


J = 7
K = 16
L = 21

mu_intercept = 8309

# from elsewhere - calculation run on full data
mu_year_effect = -800
sd_year_effect = 800

###MODEL_OBJECT####
# construct model object
npp_model <- stan_model(file = "C:/Users/Caleb/Desktop/479/Project/stan_files/npp_model2.stan")

###FIT_MODEL####
npp_data <- list(J=J, K=K, L=L, 
                 mu_intercept = mu_intercept, 
                 zone_mu_coefs = zone.coefs,
                 type_mu_coefs = type.coefs,
                 year_mu_coefs = year.coefs,
                 sd_adj = sd.mat,
                 dat=dat,
                 mu_year_effect = mu_year_effect,
                 sd_year_effect = sd_year_effect
                 )

npp_fit <- sampling(object = npp_model,
                    data = npp_data,
                    iter=2000, 
                    chains=4)
#pairs(npp_fit)
###SUMMARY####
npp_summary <- summary(npp_fit)

tmp <- npp_summary$summary

###EXTRACT_SAMPLES###
str(extract(npp_fit))
samples1 <- extract(npp_fit)[[1]]

# extract samples for a particular param:
year_effect_samples <- extract(npp_fit, par="year_effect")[["year_effect"]]; str(year_effect_samples)
mu_adj_samples <- extract(npp_fit, par="mu_adj")[["mu_adj"]]; str(mu_adj_samples)
y_post_samples <- extract(npp_fit, par="y_post")[["y_post"]]

# year 1 effect
hist(year_effect_samples[,1], main="Samples of Year 2000 Effect")

# Aspen Forested Wetland in zone 1 samples
hist(mu_adj_samples[,1,1,1], main="Samples of Median NPP: Aspen in Zone 1, Year 2000")

#

###POSTERIOR_PREDICT####
post_pred_v <- extract(npp_fit, par="y_post")[["y_post"]]

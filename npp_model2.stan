// Stan code for model:

// data block - includes prior hyperparameters
data {
  // data:
  int<lower=0> J; // number of types - expecting 7
  int<lower=0> K; // number of zones - expecting 16
  int<lower=0> L; // number of years - expecting 21
  
  real mu_intercept; // mu intercept value
  
  real zone_mu_coefs[K]; // zone mu coefficients
  real type_mu_coefs[J]; // type mu coefficients
  real year_mu_coefs[L]; // year mu coefficients
  
  real sd_adj[J, K]; // array of adjusted sd vals - have data everywhere
  
  //real<lower=0> sd_intercept; // sd intercept value
  //real zone_sd_coefs[K]; // sd zone coefs
  //real type_sd_coefs[J]; // sd type coefs
  
  real dat[J, K, L]; // array of NPP median data - types, zone, years
  // contains -1's where no data actually present
  
  // hyper-parameters:
  real mu_year_effect; // year effect mean
  real<lower=0> sd_year_effect; // year effect sd
}

// parameter block - quantities we want a posterior estimate for
parameters {
  // global NPP params
  real year_effect[L];
}

// transformed parameters block
// transform year effect into mu_adj
transformed parameters {
  real year_effect_tf[L]; // vector to store year effects
  real mu_adj[J, K, L]; // matrix to store mean values - -1 where no data
  // compose mu_adj
  // loop for year
  for (l in 1:L) {
    year_effect_tf[l] = year_effect[l]; // for each year, pull year effect
    // loop for zones
    for (k in 1:K) {
      // loop for types
      for(j in 1:J) {
        // if data[j,k,l] = -1, continue
        if (dat[j,k,l] < 0) {continue;}
        //otherwise add relevant regression coefficients to year effect
        mu_adj[j,k,l] = mu_intercept + year_effect_tf[l] + zone_mu_coefs[k] + type_mu_coefs[j]; 
      }
    }
  }
}

// model block - steps to model eventual response of interest
model {
  // loop for years
  for(l in 1:L) {
    // each year : year effect from year prior dist
    year_effect[l] ~ normal(mu_year_effect, sd_year_effect);
    // loop for zones
    for(k in 1:K) {
      // loop for types
      for(j in 1:J) {
        if(dat[j,k,l] < 0) {continue;}
        // yearly region-type median npp drawn from normal according to:
        dat[j, k, l] ~ normal(mu_adj[j,k,l], sd_adj[j,k]);
      }
    }
  }
}

generated quantities {
  real y_post[J, K, L]; // array to store posterior predictions
  // loop for years
  for(l in 1:L) {
    // loop for zones
    for(k in 1:K) {
      // loop for types
      for(j in 1:J) {
        // if we don't actually have data, continue
        if(dat[j,k,l] < 0) {continue;}
        y_post[j,k,l] = normal_rng(mu_adj[j,k,l], sd_adj[j,k]);
      }
    }
  }
}


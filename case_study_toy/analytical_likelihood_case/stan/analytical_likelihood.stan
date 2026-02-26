data {
  int<lower=3> N;                 // number of nodes per graph
  array[N] int<lower=0, upper=1> x;              // group labels
  array[N, N] int<lower=0, upper=1> adj;         // adjacency matrices
}

parameters {
  real<lower=0.1, upper=0.9> pi_aa;
  real<lower=0.1, upper=0.9> pi_bb;
  real<lower=0.1, upper=0.9> pi_ab;
}

model {
  // priors (matching your uniform draws)
  pi_aa ~ uniform(0.1, 0.9);
  pi_bb ~ uniform(0.1, 0.9);
  pi_ab ~ uniform(0.1, 0.9);

  // likelihood
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      real p;
      if (x[i] == x[j]) {
        p = (x[i] == 1) ? pi_aa : pi_bb;
      } else {
        p = pi_ab;
      }
      target += bernoulli_lpmf(adj[i, j] | p);
    }
  }
}


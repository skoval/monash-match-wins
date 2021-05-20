functions{
real calculate_win(int i, int[] player1_indexes, int[] player0_indexes, real[] lambda, matrix h2h) //parameters
{
    real log_mean = lambda[player1_indexes[i]] - lambda[player0_indexes[i]] + h2h[player1_indexes[i], player0_indexes[i]];

return inv_logit(log_mean);
}

}

data {
  int <lower=1> N_total; // Sample size
  int <lower=0, upper=1> y[N_total]; //variable that indicates which one wins player0 or player1
  int <lower=1> N_players; // Number of players
  int <lower=1> player0_indexes[N_total];
  int <lower=1> player1_indexes[N_total];
}

parameters {
  real<lower=0> prior_lambda_std;
  real lambda[N_players]; //Latent variable that represents the strength
  real<lower=0> h2h_std; // Random effects for h2h
  matrix[N_players,N_players] h2h_z;
}

transformed parameters {
  matrix[N_players,N_players] h2h;
  
  for(i in 1:N_players)
    for(j in 1:N_players){
      if(i == j)
        h2h[i,j] = 0;
      else if(i > j)
        h2h[i,j] = -1 * h2h_z[j,i] * h2h_std; // Take symmetrical position of opposite sign
      else
        h2h[i,j] = h2h_z[i,j] * h2h_std; // Assign random effect for head-to-head
    }
}

model {
  //priors
  prior_lambda_std ~ std_normal();
  lambda ~ normal(0.,prior_lambda_std); // center at 0

  h2h_std ~ std_normal();

  for(i in 1:N_players)
    for(j in 1:N_players){
      if(i >= j)
        h2h_z[i,j] ~ normal(0, .0001); // Effective zero assignment
      else
        h2h_z[i,j] ~ std_normal();
    }

  //model
  for (i in 1:N_total)
  {
    real p1_win = calculate_win(i, player1_indexes, player0_indexes, lambda, h2h);
    target += bernoulli_lpmf(y[i] | p1_win);
  }
}

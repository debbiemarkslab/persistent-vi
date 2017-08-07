#ifndef BAYES_H
#define BAYES_H

/* Defines numeric_t */
#include "pvi.h"

/* Hamiltonian Monte Carlo */
typedef numeric_t (*hmc_hfun_t) (void *data, const numeric_t *x, const int n);
typedef void (*hmc_gradfun_t) (void *data, const numeric_t *x, numeric_t *g,
    const int n);
numeric_t SampleHamiltonianMonteCarlo(hmc_hfun_t hfun, hmc_gradfun_t grad, 
    void *data, numeric_t *X, int n, int s, int L, numeric_t *epsilon,
    int warmup);

/* Gaussian stochastic variational inference */
typedef numeric_t (*neglogp_t) (void *data, const numeric_t *x,
	numeric_t *g, const int n);
numeric_t EstimateGaussianVariationalApproximation(neglogp_t neglogp,
    void *data, numeric_t *mu, numeric_t *sigma, int n, int k, numeric_t eps,
    int maxIter, numeric_t crit);

/* MAP estimation by SGD (Adam) */
typedef void (*gradfun_t) (void *data, const numeric_t *x, numeric_t *g,
    const int n);
void EstimateMaximumAPosteriori(gradfun_t gradlogp, void *data,
    numeric_t *x, int n, numeric_t eps, int maxIter, numeric_t crit);

/* Bayesian approaches to categorical distributions */
void EstimateCategoricalDistribution(const numeric_t *C, numeric_t *P, int n);
void SampleCategoricalDistribution(const numeric_t *C, numeric_t *P, int n);

/* Random Number Generation */
/* Initialization */
void InitRNG(int seed);
int RandomInt(int N);
numeric_t RandomUniform();
numeric_t RandomNormal();
numeric_t RandomGamma(numeric_t alpha);

numeric_t QuickSelect(numeric_t *A, int len, int k);
#endif /* BAYES_H */
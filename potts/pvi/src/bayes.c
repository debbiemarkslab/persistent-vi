#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>

#include "include/pvi.h"
#include "include/twister.h"
#include "include/bayes.h"

#define PI 3.14159265358979323846

/* Internal prototypes */
numeric_t ElapsedTime(struct timeval *start);

numeric_t EstimateGaussianVariationalApproximation(neglogp_t neglogp,
    void *data, numeric_t *mu, numeric_t *sigma, int n, int k,
    numeric_t eps, int maxIter, numeric_t crit) {
    /* Estimate a diagonal Gaussian variational approximation Q of a
       distribution P by stochastically minimizing KL(Q||P)
       (maximizing the ELBO)
       Arguments:
            neglogp         value and gradient of -log(P(params|data)) 
            data            pointer to data
            mu              estimated means (length n)
            sigma           estimated standard deviations (length n)
            n               number of parameters
            k               number of samples for each estimate of grad(KL)
            eps             learning rate
            maxIter         maximum number of iterations
            crit            stop when ||grad|| / ||x|| < crit
     */

    /* Use the Mersenne Twister for reproducible sampling results */
    init_genrand(42);

    /* Parameterize variance parameters by their logarithms */
    numeric_t *logSig = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) logSig[i] = log(sigma[i]);

    /* Initialize Q ~ Norm(mu, exp(2*logSig)) with a small spherical Gaussian */
    // for (int i = 0; i < n; i++) mu[i] = 0;
    // for (int i = 0; i < n; i++) logSig[i] = 0;

    /* ELBO and its gradients */
    numeric_t negELBO = 0;
    numeric_t *gradMu = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *gradLogSig = (numeric_t *) malloc(n * sizeof(numeric_t));    

    /* Use a vector of standard normals Z to sample S from Q */
    numeric_t *z = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *s = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *gradP = (numeric_t *) malloc(n * sizeof(numeric_t));

    /* Stopping criteria */
    numeric_t meanELBO = 0;
    numeric_t meanLogP = 0;
    numeric_t criterion = crit + 1.0;

    /* Begin profiling */
    struct timeval start;
    gettimeofday(&start, NULL);

    /* -------------- Stochastically maximize the ELBO by Adam -------------- */
    /* Initialize estimates of first and second moments of the gradient */
    numeric_t *meanGradMu = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *meanGradLogSig = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *squareGradMu = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *squareGradLogSig = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) meanGradMu[i] = 0;
    for (int i = 0; i < n; i++) meanGradLogSig[i] = 0;
    for (int i = 0; i < n; i++) squareGradMu[i] = 0;
    for (int i = 0; i < n; i++) squareGradLogSig[i] = 0;
    int t = 1;
    do {
        /* Estimate the ELBO and its gradient by averaging k samples from Q */
        numeric_t negLogP = 0;
        for (int i = 0; i < n; i++) gradMu[i] = 0;
        for (int i = 0; i < n; i++) gradLogSig[i] = 0;
        for (int i = 0; i < k; i++) {
            /* Sample S from current Q */
            for (int j = 0; j < n; j++) z[j] = RandomNormal();
            for (int j = 0; j < n; j++)
                s[j] = mu[j] + exp(logSig[j]) * z[j];

            /* Cross-entropy term, logP(S|data) */
            negLogP += neglogp(data, s, gradP, n);

            /* Contribute dlogP(S|data)/dMu & dlogP(S|data)/dLogSig */
            for (int j = 0; j < n; j++) gradMu[j] += gradP[j];
            for (int j = 0; j < n; j++)
                gradLogSig[j] += gradP[j] * z[j] * exp(logSig[j]);
        }
        numeric_t invK = 1.0 / ((numeric_t) k);
        negLogP *= invK;
        for (int i = 0; i < n; i++) gradMu[i] *= invK;
        for (int i = 0; i < n; i++) gradLogSig[i] *= invK;

        /* Entropy term E[logQ(S)] is analytic */
        numeric_t entropy = n * 0.5 * log(2 * PI * exp(1));
        for (int i = 0; i < n; i++) entropy += logSig[i];
        negELBO = negLogP - entropy;
        for (int i = 0; i < n; i++) gradLogSig[i] -= 1.0;

        /* Update estimates of moments */
        numeric_t beta1 = 0.9;
        numeric_t beta2 = 0.999;
        for (int i = 0; i < n; i++)
            meanGradMu[i] = beta1 * meanGradMu[i] 
                          + (1.0 - beta1) * gradMu[i];
        for (int i = 0; i < n; i++)
            meanGradLogSig[i] = beta1 * meanGradLogSig[i] 
                              + (1.0 - beta1) * gradLogSig[i];
        for (int i = 0; i < n; i++)
            squareGradMu[i] = beta2 * squareGradMu[i] 
                            + (1.0 - beta2) * gradMu[i] * gradMu[i];
        for (int i = 0; i < n; i++)
            squareGradLogSig[i] = beta2 * squareGradLogSig[i] 
                                + (1.0 - beta2) * gradLogSig[i] * gradLogSig[i];

        /* Update Q with Adam learning rates */
        numeric_t alpha = eps * sqrt(1.0 - pow(beta2, (numeric_t) t)) 
                              / (1.0 - pow(beta1, (numeric_t) t));
        for (int i = 0; i < n; i++)
            mu[i] -= meanGradMu[i] * alpha / (sqrt(squareGradMu[i]) + 1E-8); 
        for (int i = 0; i < n; i++)
            logSig[i] -= meanGradLogSig[i]
                         *  alpha / (sqrt(squareGradLogSig[i]) + 1E-8);

        /* Stopping criterion: ||grad(params)|| / ||params|| */
        numeric_t paramNorm = 1E-6;
        for (int i = 0; i < n; i++)
            paramNorm += mu[i] * mu[i]; 
        for (int i = 0; i < n; i++)
            paramNorm += logSig[i] * logSig[i];
        numeric_t gradNorm = 1E-6;
        for (int i = 0; i < n; i++)
            gradNorm += meanGradMu[i] * meanGradMu[i] / squareGradMu[i]; 
        for (int i = 0; i < n; i++)
            gradNorm += meanGradLogSig[i] * meanGradLogSig[i] / squareGradLogSig[i]; 
        paramNorm = sqrt(paramNorm);
        gradNorm = sqrt(gradNorm);
        criterion = gradNorm / paramNorm;

        /* Compute exponentially averaged ELBO and LogP */
        numeric_t rate = 0.02;
        if (t == 1) {
            meanELBO = -negELBO;
            meanLogP = -negLogP;
        } else {
            meanELBO = (1.0 - rate) * meanELBO - rate * negELBO;
            meanLogP = (1.0 - rate) * meanLogP - rate * negLogP;
        }

        if (t == 1)
            fprintf(stderr, "iter\ttime\tELBO\t\tLogP\t\t||x||\t||g||\tcrit\n");
        fprintf(stderr, "%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.3f\n",
            t, ElapsedTime(&start), meanELBO, meanLogP, paramNorm, gradNorm,
            criterion);

        t++;
    } while (t <= maxIter && criterion > crit);
    free(meanGradMu);
    free(meanGradLogSig);
    free(squareGradMu);
    free(squareGradLogSig);

    /* Transform back to linear-space sigmas */
    for (int i = 0; i < n; i++) sigma[i] = exp(logSig[i]);

    free(logSig);
    free(s);
    free(z);
    free(gradP);
    free(gradMu);
    free(gradLogSig);

    return -negELBO;
}

void EstimateMaximumAPosteriori(gradfun_t gradlogp, void *data,
    numeric_t *x, int n, numeric_t eps, int maxIter, numeric_t crit) {
    /* Compute a MAP (Maximum A Posteriori) estimate of the posterior
       distribution P(x|data) by Stochastic Gradient Descent (Adam)
       Arguments:
            gradlogp        gradient of -log(P(params|data)) 
            data            pointer to data
            x               estimated parameters (length n)
            n               number of parameters
            eps             learning rate
            maxIter         maximum number of iterations
            crit            stop when ||grad|| / ||x|| < crit
     */
    /* Use the Mersenne Twister for reproducible sampling results */
    init_genrand(42);

    numeric_t *g = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t criterion = crit + 1.0;

    /* Begin profiling */
    struct timeval start;
    gettimeofday(&start, NULL);

    /* --------------- Stochastically minimize -logP by Adam ---------------- */
    /* Initialize estimates of first and second moments of the gradient */
    numeric_t *meanG = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *squareG = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) meanG[i] = 0;
    for (int i = 0; i < n; i++) squareG[i] = 0;
    int t = 1;
    do {
        /* Estimate the gradient */
        for (int i = 0; i < n; i++) g[i] = 0;
        gradlogp(data, x, g, n);

        /* Update estimates of moments */
        numeric_t beta1 = 0.99;
        numeric_t beta2 = 0.999;
        for (int i = 0; i < n; i++)
            meanG[i] = beta1 * meanG[i] + (1.0 - beta1) * g[i];
        for (int i = 0; i < n; i++)
            squareG[i] = beta2 * squareG[i] + (1.0 - beta2) * g[i] * g[i];

        /* Update Q with Adam learning rates */
        numeric_t schedule = (1.0 - ((numeric_t) t) / ((numeric_t) maxIter));
        // numeric_t schedule = (1.0 / pow(t, 0.501));
        numeric_t alpha = eps * schedule
                              * sqrt(1.0 - pow(beta2, (numeric_t) t)) 
                                  / (1.0 - pow(beta1, (numeric_t) t));
        for (int i = 0; i < n; i++)
            x[i] -= meanG[i] * alpha / (sqrt(squareG[i]) + 1E-8);

        /* Stopping criterion: ||grad(params)|| / ||params|| */
        numeric_t paramNorm = 1E-6;
        for (int i = 0; i < n; i++) paramNorm += x[i] * x[i];
        numeric_t gradNorm = 1E-6;
        for (int i = 0; i < n; i++)
            gradNorm += meanG[i] * meanG[i] / (squareG[i] + 1E-8);
        paramNorm = sqrt(paramNorm);
        gradNorm = sqrt(gradNorm);
        criterion = gradNorm / paramNorm;

        if (t == 1)
            fprintf(stderr, "iter\ttime\t||x||\t||g||\tcrit\n");
        fprintf(stderr, "%d\t%.1f\t%.1f\t%.1f\t%.1f\n",
            t, ElapsedTime(&start), paramNorm, gradNorm, criterion);
        t++;
    } while (t <= maxIter && criterion > crit);
    free(meanG);
    free(squareG);
    free(g);
}

numeric_t SampleHamiltonianMonteCarlo(hmc_hfun_t hfun, hmc_gradfun_t grad,
    void *data, numeric_t *X, int n, int s, int L, numeric_t *epsilon,
    int warmup) {
    /* Draw samples X from a pdf specified by P(params | data) using
       Hamiltonian Monte Carlo */

    /* Initialize position(x), gradient(g), and momentum(p) */
    numeric_t *x = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *xnew = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *g = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *gnew = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *p = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) x[i] = 0;
    for (int i = 0; i < n; i++) xnew[i] = 0;
    for (int i = 0; i < n; i++) g[i] = 0;
    for (int i = 0; i < n; i++) gnew[i] = 0;
    for (int i = 0; i < n; i++) p[i] = 0;
    numeric_t U = hfun(data, x, n);
    grad(data, x, g, n);

    /* Use the Mersenne Twister for reproducible sampling results */
    init_genrand(42);

    /* Step sizes are log-normally distribution with median EPS */
    numeric_t eps = *epsilon;
    numeric_t epsMu = log(eps);
    numeric_t epsStd = log(4.0);


    /* Store running averages of the statistics for mass tuning */
    numeric_t *xMean = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *xxMean = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) xMean[i] = 0;
    for (int i = 0; i < n; i++) xxMean[i] = 0;
    int xCount = 0;

    /* Mass matrix (diagonal) scales the step sizes along each dimension */
    numeric_t *invMass = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *massStd = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) invMass[i] = 1.0;
    for (int i = 0; i < n; i++) massStd[i] = 1.0;

    /* --------------- Warmup steps to estimate the mass matrix ------------- */
    int massStart = warmup / 3;
    int massEnd = 2 * warmup / 3;
    int adaptStart = 0;
    int adaptEnd = warmup;
    int wsteps = 0;
    /* Lagged number of accepted moves over past 10 proposals */
    int acceptCount = 0;
    int windowCount = 0;
    do {
        /* Randomize momentum */
        for (int i = 0; i < n; i++) p[i] = RandomNormal() * massStd[i];
        numeric_t K = 0;
        for (int i = 0; i < n; i++) K += p[i] * p[i] * invMass[i];
        K *= 0.5;
        numeric_t H = U + K;

        /* Integrate system by L leapfrog steps */
        eps = exp(epsStd * RandomNormal() + epsMu);
        for (int i = 0; i < n; i++) xnew[i] = x[i];
        for (int i = 0; i < n; i++) gnew[i] = g[i];
        for (int lx = 0; lx < L; lx++) {
            for (int i = 0; i < n; i++) p[i] -= eps * gnew[i] * 0.5;
            for (int i = 0; i < n; i++) xnew[i] += eps * p[i] * invMass[i];
            grad(data, xnew, gnew, n);
            for (int i = 0; i < n; i++) p[i] -= eps * gnew[i] * 0.5;
        }

        /* Evaluate the proposal */
        numeric_t Unew = hfun(data, xnew, n);
        numeric_t Knew = 0;
        for (int i = 0; i < n; i++) Knew += p[i] * p[i] * invMass[i];
        Knew *= 0.5;
        numeric_t Hnew = Unew + Knew;
        numeric_t dH = Hnew - H;
        if (dH < 0 || ((numeric_t) genrand_real3()) < exp(-dH)) {
            /* Accept the proposal */
            U = Unew;
            for (int i = 0; i < n; i++) x[i] = xnew[i];
            for (int i = 0; i < n; i++) g[i] = gnew[i];
            wsteps++;
            acceptCount++;

            /* Store running averages of the statistics for mass tuning */
            if (wsteps >= massStart && wsteps <= massEnd) {
                for (int i = 0; i < n; i++) xMean[i] += x[i];
                for (int i = 0; i < n; i++) xxMean[i] += x[i] * x[i];
                xCount++;
            }

            /* Use the sample variance of each variable for inverse mass */
            if (wsteps == massEnd) {
                numeric_t invCounts = 1.0 / (numeric_t) xCount;
                for (int i = 0; i < n; i++) xMean[i] *= invCounts;
                for (int i = 0; i < n; i++) xxMean[i] *= invCounts;
                for (int i = 0; i < n; i++)
                    invMass[i] = xxMean[i] - xMean[i] * xMean[i] + 0.01;
                for (int i = 0; i < n; i++)
                    massStd[i] = sqrt(1.0 / invMass[i]);
            }
        }

        /* Radford Neal's adaptation scheme for step size */
        if (wsteps >= adaptStart && wsteps <= adaptEnd) {
            windowCount++;
            if (windowCount == 10) {
                if (acceptCount == 0) {
                    /* If last 10 were rejected, decrease eps by 20% */
                    epsMu += log(0.8);
                } else if (acceptCount > 8) {
                    /* If >8 were accepted, increase eps by 20% */
                    epsMu += log(1.2);
                }
                acceptCount = 0;
                windowCount = 0;
            }
        }
    } while (wsteps < warmup);

    /* Tighten the step size variation during sampling */
    *epsilon = exp(epsMu);
    epsStd = log(2.0);

    /* --------------------------- Sampling steps --------------------------- */
    int sx = 0;
    int nsteps = 0;
    do {
        /* Randomize momentum */
        for (int i = 0; i < n; i++) p[i] = RandomNormal() * massStd[i];
        numeric_t K = 0;
        for (int i = 0; i < n; i++) K += p[i] * p[i] * invMass[i];
        K *= 0.5;
        numeric_t H = U + K;

        /* Integrate system by L leapfrog steps */
        eps = exp(epsStd * RandomNormal() + epsMu);
        for (int i = 0; i < n; i++) xnew[i] = x[i];
        for (int i = 0; i < n; i++) gnew[i] = g[i];
        for (int lx = 0; lx < L; lx++) {
            for (int i = 0; i < n; i++) p[i] -= eps * gnew[i] * 0.5;
            for (int i = 0; i < n; i++) xnew[i] += eps * p[i] * invMass[i];
            grad(data, xnew, gnew, n);
            for (int i = 0; i < n; i++) p[i] -= eps * gnew[i] * 0.5;
        }

        /* Accept or reject */
        numeric_t Unew = hfun(data, xnew, n);
        numeric_t Knew = 0; 
        for (int i = 0; i < n; i++) Knew += p[i] * p[i] * invMass[i];
        Knew *= 0.5;
        numeric_t Hnew = Unew + Knew;
        numeric_t dH = Hnew - H;
        if (dH < 0 || ((numeric_t) genrand_real3()) < exp(-dH)) {
            U = Unew;
            for (int i = 0; i < n; i++) x[i] = xnew[i];
            for (int i = 0; i < n; i++) g[i] = gnew[i];
            for (int i = 0; i < n; i++) X[sx * n + i] = xnew[i];
            sx++;
        }
        nsteps++;
    } while (sx < s && nsteps < 20 * s);

    free(x);
    free(xnew);
    free(g);
    free(gnew);
    free(p);
    free(invMass);
    free(xMean);
    free(xxMean);
    free(massStd);

    numeric_t accRate = (numeric_t) sx / (numeric_t) nsteps;
    return accRate;
}

void EstimateCategoricalDistribution(const numeric_t *C, numeric_t *P, int n) {
    /* Estimates a categorical distribution P[i] from counts C[i] with a
       symm. Dirichlet prior for P[i] and a uniform hyperprior for log(alpha) */
    /* Determine sample size */
    numeric_t N = 0;
    for (int i = 0; i < n; i++) N += C[i];
    numeric_t numA = (numeric_t) n;

    /* Midpoint quadrature over log(alpha) */
    numeric_t minAlpha = 0.01;
    numeric_t maxAlpha = 100.0;
    int numAlpha = 30;
    numeric_t *A = (numeric_t *) malloc(sizeof(numeric_t) * numAlpha);
    numeric_t *PA = (numeric_t *) malloc(sizeof(numeric_t) * numAlpha);
    for (int a = 0; a < numAlpha; a++) {
        numeric_t logAlpha = (log(maxAlpha) - log(minAlpha))
                           * (((numeric_t) a) + 0.5) / ((numeric_t) numAlpha)
                           + log(minAlpha);
        A[a] = exp(logAlpha);
        PA[a] = lgamma(numA * A[a])
                -lgamma(N + numA * A[a])
                -numA * lgamma(A[a]);
        for (int i = 0; i < n; i++) PA[a] += lgamma(A[a] + C[i]);
    }

    numeric_t scale = PA[0];
    for (int a = 0; a < numAlpha; a++) if (PA[a] > scale) scale = PA[a];
    /* Transformed distribution for log(alpha) picks up a factor of alpha */
    for (int a = 0; a < numAlpha; a++) PA[a] = A[a] * exp(PA[a] - scale);

    /* Average the conditional posterior expectations */
    for (int i = 0; i < n; i++) P[i] = 0;
    for (int a = 0; a < numAlpha; a++)
        for (int i = 0; i < n; i++)
            P[i] += PA[a] * (C[i] + A[a]) / (N + A[a] * ((numeric_t) n));

    /* Renormalize */
    numeric_t Z = 0;
    for (int i = 0; i < n; i++) Z += P[i];
    numeric_t invZ = 1.0 / Z;
    for (int i = 0; i < n; i++) P[i] *= invZ;

    free(PA);
    free(A);
}

void SampleCategoricalDistribution(const numeric_t *C, numeric_t *P, int n) {
    /* Samples a categorical distribution P[i] from counts C[i] with a
       symm. Dirichlet prior for P[i] and a uniform hyperprior for log(alpha) */

    /* Determine sample size */
    numeric_t N = 0;
    for (int i = 0; i < n; i++) N += C[i];
    numeric_t numA = (numeric_t) n;

    /* Sample log(alpha) from a histogram approximation of the posterior */
    int numAlpha = 30;
    numeric_t minAlpha = 0.01;
    numeric_t maxAlpha = 100.0;
    numeric_t *A = (numeric_t *) malloc(sizeof(numeric_t) * numAlpha);
    numeric_t *PA = (numeric_t *) malloc(sizeof(numeric_t) * numAlpha);
    /* Compute logP(alpha) for numerical stability */
    for (int a = 0; a < numAlpha; a++) {
        numeric_t logAlpha = (log(maxAlpha) - log(minAlpha))
                           * (((numeric_t) a) + 0.5) / ((numeric_t) numAlpha)
                           + log(minAlpha);
        A[a] = exp(logAlpha);
        PA[a] = lgamma(numA * A[a])
                -lgamma(N + numA * A[a])
                -numA * lgamma(A[a]);
        for (int i = 0; i < n; i++) PA[a] += lgamma(A[a] + C[i]);
    }
    numeric_t scale = PA[0];
    for (int a = 0; a < numAlpha; a++) if (PA[a] > scale) scale = PA[a];
    /* P(log(alpha)) = alpha exp(logP(alpha)) from Jacobian */
    for (int a = 0; a < numAlpha; a++) PA[a] = A[a] * exp(PA[a] - scale);
    /* Convert unnormalized histogram to unnormalized CDF */
    for (int a = 1; a < numAlpha; a++) PA[a] = PA[a] + PA[a - 1];

    /* Sample alpha from CDF given C[i] */
    numeric_t U = RandomUniform() * PA[numAlpha - 1];
    int ax = 0;
    while (U > PA[ax]) ax++;
    numeric_t Asamp = A[ax];

    /* Sample P[i] from Dirichlet given alpha, C[i] */
    for (int i = 0; i < n; i++) P[i] = RandomGamma(C[i] + A[ax]);
    numeric_t Z = 0;
    for (int i = 0; i < n; i++) Z += P[i];
    numeric_t invZ = 1.0 / Z;
    for (int i = 0; i < n; i++) P[i] *= invZ;

    free(PA);
    free(A);
}

/* ---------------------------------_DEBUG_---------------------------------- */
/* Test random number generators */
// FILE *fpOutput = fopen("normal.txt", "w");
// for (int i = 0; i < 1E6; i++) fprintf(fpOutput, "%f\n", RandomNormal());
// fclose(fpOutput);
// fpOutput = fopen("gamma0.5.txt", "w");
// for (int i = 0; i < 1E6; i++) fprintf(fpOutput, "%f\n", RandomGamma(0.5));
// fclose(fpOutput);
// fpOutput = fopen("gamma1.txt", "w");
// for (int i = 0; i < 1E6; i++) fprintf(fpOutput, "%f\n", RandomGamma(1.0));
// fclose(fpOutput);
// fpOutput = fopen("gamma2.txt", "w");
// for (int i = 0; i < 1E6; i++) fprintf(fpOutput, "%f\n", RandomGamma(2.0));
// fclose(fpOutput);
// exit(0);
/* ---------------------------------^DEBUG^---------------------------------- */

void InitRNG(int seed) {
    /* Initializes the Mersenne Twister in twister.c */
    init_genrand(seed);
}

int RandomInt(int N) {
    /* Generates a random integer on [0, N-1] */
    return genrand_int32() % N;
}

numeric_t RandomUniform() {
    /* Generates a standard uniform on (0,1). This wraps the Mersenne Twister in
       twister.c */
    return (numeric_t) genrand_real3();
}

numeric_t RandomNormal() {
    /* Generates a standard normal with zero mean and unit variance by the 
       polar method (Marsaglia & Bray 1964) */
    numeric_t u, v, s;
    do {
        u = RandomUniform() * 2.0 - 1.0;
        v = RandomUniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0);
    numeric_t mul = sqrt(-2.0 * log(s) / s);
    return u * mul;
}

numeric_t RandomGamma(numeric_t alpha) {
    /* Generates a standard gamma */
    if (alpha >= 1.0) {
        /* Rejection-transformation method of (Marsaglia & Tsang, 2000) */ 
        numeric_t d = alpha - 1.0 / 3.0;
        numeric_t c = 1.0 / sqrt(9 * d);
        numeric_t x, u, v;
        do {
            x = RandomNormal();
            u = RandomUniform();
            v = 1.0 + x * c;
            v = v * v * v;
        } while ((v <= 0) || (log(u) >= 0.5 * x * x + d - d * v + d * log(v)));
        return d * v;
    } else {
        /* Rejection method of (Kundu & Gupta, 2007) */
        numeric_t c = 1.0 / alpha;
        numeric_t u, v, x, ex;
        do {
            u = RandomUniform();
            v = RandomUniform();
            x = -2.0 * log(1.0 - pow(u, c));
            ex = exp(-x / 2.0);
        } while (v > ex * pow(0.5 * x / (1.0 - ex), alpha - 1.0));
        return x;
    }
}

numeric_t QuickSelect(numeric_t *A, int len, int k) {
/* Returns the kth smallest element of A, for use in quantile estimates
   The xth percentile can be estimated by k = floor(x/100 * len) */
    int pivot = 0;
    /* Swap from left end until the pivot is smaller */
    for (int i = 0; i < len - 1; i++) {
        if (A[i] > A[len - 1]) continue;
        numeric_t temp = A[i];
        A[i] = A[pivot];
        A[pivot] = temp;
        pivot++;
    }

    /* Swap the pivot to right end */
    numeric_t temp = A[len - 1];
    A[len - 1] = A[pivot];
    A[pivot] = temp;

    if (k == pivot) {
        return A[pivot];
    } else if (pivot > k) {
        return QuickSelect(A, pivot, k);
    } else {
        return QuickSelect(&(A[pivot]), len - pivot, k - pivot);
    }
}

numeric_t ElapsedTime(struct timeval *start) {
/* Computes the elapsed time from START to NOW in seconds */
    struct timeval now;
    gettimeofday(&now, NULL);
    if (now.tv_usec < start->tv_usec) {
        int nsec = (start->tv_usec - now.tv_usec) / 1000000 + 1;
        start->tv_usec -= 1000000 * nsec;
        start->tv_sec += nsec;
    }
    if (now.tv_usec - start->tv_usec > 1000000) {
        int nsec = (now.tv_usec - start->tv_usec) / 1000000;
        start->tv_usec += 1000000 * nsec;
        start->tv_sec -= nsec;
    }
    return (numeric_t) (now.tv_sec - start->tv_sec)
                      + ((numeric_t) (now.tv_usec - start->tv_usec)) / 1E6;
}
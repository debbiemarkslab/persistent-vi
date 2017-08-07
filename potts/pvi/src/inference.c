#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>

/* Optionally include OpenMP with the -fopenmp flag */
#if defined(_OPENMP)
    #include <omp.h>
#endif

/* Optimization and inference libraries */
#include "include/lbfgs.h"
#include "include/twister.h"
#include "include/bayes.h"

#include "include/pvi.h"
#include "include/inference.h"

#define PI 3.14159265358979323846

/* Numerical bounds for ZeroAPCPriors */
#define LAMBDA_J_MIN 1E-2
#define LAMBDA_J_MAX 1E4
#define REGULARIZATION_GROUP_EPS 1E-6

/* Internal to InferPairModel: 
   Bayesian estimation of hyperparameters for sites by MCMC (HMC) */
void EstimateSiteLambdasBayes(numeric_t *lambdas, alignment_t *ali,
    options_t *options);
/* Internal to EstimateSiteLambdasBayes: Hierarchical model for single sites */
numeric_t HMCSiteH(void *data, const numeric_t *x, const int n);
void HMCSiteHGrad(void *data, const numeric_t *x, numeric_t *g, const int n);
numeric_t HMCSiteHNonCenter(void *data, const numeric_t *x, const int n);
void HMCSiteHGradNonCenter(void *data, const numeric_t *x, numeric_t *g, 
    const int n);

/* Internal to InferPairModel:
   Bayesian estimation of parameters by a gaussian variational approximation */
void EstimatePairModelVBayes(numeric_t *x, numeric_t *lambdas, alignment_t *ali, 
    options_t *options);
/* Internal to EstimatePairModelBayes: Hierarchical model */
numeric_t VBayesPairHierarchicalNonCentGibbs(void *data, const numeric_t *xB,
    numeric_t *gB, const int n);
numeric_t VBayesPairHierarchicalNonCentPL(void *data, const numeric_t *xB,
    numeric_t *gB, const int n);
numeric_t VBayesSiteHierarchicalNonCent(void *data, const numeric_t *xB, 
    numeric_t *gB, const int n);
numeric_t VBayesPairHierarchical(void *data, const numeric_t *xB, numeric_t *gB,
    const int n);

/* Internal to InferPairModel: 
   Bayesian estimation of parameters by MCMC (HMC) */
void EstimatePairModelHMC(numeric_t *lambdas, numeric_t *x, alignment_t *ali, 
    options_t *options);
/* Internal to EstimatePairModelBayes: Hierarchical model */
numeric_t HMCPairHNonCenter(void *data, const numeric_t *xB, const int n);
void HMCPairHGradNonCenter(void *data, const numeric_t *xB, numeric_t *gB, 
    const int n);

/* Internal to InferPairModel: 
    MAP estimation of parameters by stochastic Maximum Likelihood */
void EstimatePairModelMAP(numeric_t *x, numeric_t *lambdas, alignment_t *ali,
    options_t *options);
/* Internal to EstimatePairModelMAP:
    stochastic gradient esimators */
void MAPPairGibbs(void *data, const numeric_t *x, numeric_t *g, const int n);

/* Internal to InferPairModel: 
    MAP estimation of parameters by L-BFGS */
void EstimatePairModelPLM(numeric_t *x, numeric_t *lambdas, alignment_t *ali,
    options_t *options);
/* Internal to EstimatePairModelPLM: 
   Objective functions for point parameter estimates (MAP) */
static lbfgsfloatval_t PLMNegLogPosterior(void *instance,
    const lbfgsfloatval_t *xB, lbfgsfloatval_t *gB, const int n,
    const lbfgsfloatval_t step);
static lbfgsfloatval_t PLMNegLogPosteriorGapReduce(void *instance,
    const lbfgsfloatval_t *xB, lbfgsfloatval_t *gB, const int n,
    const lbfgsfloatval_t step);
static lbfgsfloatval_t PLMNegLogPosteriorBlock(void *instance,
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n,
    const lbfgsfloatval_t step);
static lbfgsfloatval_t PLMNegLogPosteriorDO(void *instance,
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n,
    const lbfgsfloatval_t step);
/* Internal to EstimatePairModelPLM: progress reporting */
static int ReportProgresslBFGS(void *instance, const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls);
/* Internal to EstimatePairModelPLM: parameter processing */
void PreCondition(const lbfgsfloatval_t *x, lbfgsfloatval_t *g,
    alignment_t *ali, options_t *options);
numeric_t AddPriorsCentered(const numeric_t *x, numeric_t *g, 
    const numeric_t *lambdas, numeric_t fx, alignment_t *ali,
    options_t *options);
numeric_t AddPriorsNoncentered(const numeric_t *x, numeric_t *g, 
    const numeric_t *lambdas, numeric_t *gLambdas,
    numeric_t fx, alignment_t *ali, options_t *options);
void ZeroAPCPriors(alignment_t *ali, options_t *options, numeric_t *lambdas,
    lbfgsfloatval_t *x);
/* Internal to EstimatePairModelPLM: utility functions to L-BFGS */
const char *LBFGSErrorString(int ret);


numeric_t *InferPairModel(alignment_t *ali, options_t *options) {
    /* Estimate the parameters of a maximum entropy model for a
       multiple sequence alignment */

    /* Initialize the regularization parameters */
    numeric_t *lambdas =
    (numeric_t *) malloc((ali->nSites + ali->nSites * (ali->nSites - 1) / 2)
            * sizeof(numeric_t));
    for (int i = 0; i < ali->nSites; i++) lambdaHi(i) = options->lambdaH;
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            lambdaEij(i, j) = options->lambdaE;

    /* For gap-reduced problems, eliminate the gaps and reduce the alphabet */
    if (options->estimatorMAP == INFER_MAP_PLM_GAPREDUCE) {
        ali->nCodes = strlen(ali->alphabet) - 1;
        for (int i = 0; i < ali->nSites; i++)
            for (int s = 0; s < ali->nSeqs; s++)
                seq(s, i) -= 1;
    }

    /* Initialize parameters */
    ali->nParams = ali->nSites * ali->nCodes
        + ali->nSites * (ali->nSites - 1) / 2 * ali->nCodes * ali->nCodes;
    numeric_t *x = (numeric_t *) malloc(sizeof(numeric_t) * ali->nParams);
    if (x == NULL) {
        fprintf(stderr,
            "ERROR: Failed to allocate a memory block for variables.\n");
        exit(1);
    }
    for (int i = 0; i < ali->nParams; i++) x[i] = 0.0;

    /* Initialize site parameters with the ML estimates 
        hi = log(fi) + C
        A single pseudocount is added for stability 
       (Laplace's rule or Morcos et al. with lambda = nCodes) */
    numeric_t pseudoC = (numeric_t) ali->nCodes;
    numeric_t Zinv = 1.0 / (ali->nEff + pseudoC);
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nSites; ai++)
            xHi(i, ai) = Zinv * pseudoC / (numeric_t) ali->nCodes;
    for (int s = 0; s < ali->nSeqs; s++)
        for (int i = 0; i < ali->nSites; i++)
            xHi(i, seq(s, i)) += ali->weights[s] * Zinv;
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            xHi(i, ai) = log(xHi(i, ai));
    /* Zero-sum gauge */
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t hSum = 0.0;
        for (int ai = 0; ai < ali->nCodes; ai++) hSum += xHi(i, ai);
        numeric_t hShift = hSum / (numeric_t) ali->nCodes;
        for (int ai = 0; ai < ali->nCodes; ai++)
            xHi(i, ai) -= hShift;
    }

    switch(options->estimator) {
        /* Full posterior */
        case INFER_VBAYES:
            /* Approximate full posterior by a diagonal Gaussian */
            free(x);
            int nBayes = 2 * (ali->nParams
                          + 2 + ali->nSites 
                          + ali->nSites * (ali->nSites - 1) / 2);
            numeric_t *xB = (numeric_t *) malloc(sizeof(numeric_t) * nBayes);
            for (int i = 0; i < nBayes; i++) xB[i] = 0.0;
            EstimatePairModelVBayes(xB, lambdas, ali, options);
            x = xB;
            break;
        /* Point estimates */
        case INFER_MAP:
            /* Maximum a posteriori estimates of model parameters */
            EstimatePairModelMAP(x, lambdas, ali, options);
            break;
        case INFER_PLM:
            /* Maximum a posteriori estimates of model parameters */
            if (options->noncentered) {
                int offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2;
                int nNoncent = offset + ali->nParams;
                numeric_t *xB = (numeric_t *) malloc(sizeof(numeric_t) * nNoncent);
                for (int i = 0; i < nNoncent; i++) xB[i] = 0;
                for (int i = 0; i < ali->nParams; i++) xB[i + offset] = x[i];
                free(x);
                ali->nParams = nNoncent;
                EstimatePairModelPLM(xB, lambdas, ali, options);
                x = &(xB[offset]);
                numeric_t *lambdas = xB;
                /* Recenter parameters */
                for (int i = 0; i < ali->nSites; i++)
                    for (int ai = 0; ai < ali->nCodes; ai++)
                        xHi(i, ai) *= exp(lambdaHi(i));
                for (int i = 0; i < ali->nSites-1; i++)
                    for (int j = i + 1; j < ali->nSites; j++)
                        for (int ai = 0; ai < ali->nCodes; ai++)
                            for (int aj = 0; aj < ali->nCodes; aj++)
                                xEij(i, j, ai, aj) *= exp(lambdaEij(i, j));
            } else {
                EstimatePairModelPLM(x, lambdas, ali, options);
            }
            break;
        case INFER_BAYES:
            /* Estimate posterior means of model parameters by sampling (HMC)*/
            EstimatePairModelHMC(x, lambdas, ali, options);
            break;
        case INFER_HYBRID:
            /* Heuristic hyperparameters for MAP estimation */
            EstimateSiteLambdasBayes(lambdas, ali, options);
            EstimatePairModelPLM(x, lambdas, ali, options);
            break;
        default:
            /* Maximum a posteriori estimates of model parameters */
            EstimatePairModelPLM(x, lambdas, ali, options);
    }

    /* Restore the alignment encoding after inference */
    if (options->estimatorMAP == INFER_MAP_PLM_GAPREDUCE) {
        for (int i = 0; i < ali->nSites; i++)
            for (int s = 0; s < ali->nSeqs; s++)
                seq(s, i) += 1;
    }

    return (numeric_t *) x;
}

void EstimatePairModelVBayes(numeric_t *x, numeric_t *lambdas, alignment_t *ali, 
    options_t *options) {
    /* Estimate a variational (Gaussian) approximation to the full posterior 
       of parameters and hyperparameters of an undirected graphical model 
       with Gaussian priors over the couplings */
    void *data[2] = {(void *)ali, (void *)options};
    int n = 2 + ali->nParams
              + ali->nSites + ali->nSites * (ali->nSites - 1) / 2;
    numeric_t eps = 0.01;           /* Learning rate for SVI (Adam) */
    numeric_t crit = 1E-3;          /* Stopping criterion for ||g||/||x|| */

    /* Initialize Gibbs sampling with a random unconstrained sequences */
    init_genrand(42);
    ali->samples = (letter_t *)
        malloc(ali->nSites * options->gChains * sizeof(letter_t));
    for (int i = 0; i < ali->nSites * options->gChains; i++)
        ali->samples[i] = (genrand_int31() % ali->nCodes);

    /* Initialize with a site-independent model */
    int nInd = 1 + ali->nSites + ali->nSites * ali->nCodes;    
    numeric_t *muInd = (numeric_t *) malloc(nInd * sizeof(numeric_t));
    numeric_t *sigmaInd = (numeric_t *) malloc(nInd * sizeof(numeric_t));
    muInd[0] = -2;
    sigmaInd[0] = 0.1;
    for (int i = 0; i < nInd; i++) muInd[i] = 0;
    for (int i = 0; i < nInd; i++) sigmaInd[i] = 0.1;

    /* Stochastically optimize KL(Q||P(params|data)) for Gaussian Q */
    numeric_t ELBOInd =
        EstimateGaussianVariationalApproximation(VBayesSiteHierarchicalNonCent,
        data, muInd, sigmaInd, nInd, options->vSamples, eps, options->maxIter,
        crit);

    /* Infer a full pairwise model */
    numeric_t *mu = (numeric_t *) malloc(n * sizeof(numeric_t));
    numeric_t *sigma = (numeric_t *) malloc(n * sizeof(numeric_t));
    for (int i = 0; i < n; i++) mu[i] = 0;
    for (int i = 0; i < n; i++) sigma[i] = 1.0;
    mu[1] = -3;
    sigma[1] = 0.01;
    for (int i = 0; i < ali->nSites * (ali->nSites - 1) / 2; i++) {
        mu[2 + ali->nSites + i] = -2;
        sigma[2 + ali->nSites + i] = 0.1;
    }

    /* Copy relevant parameters from the site-independent model */
    mu[0] = muInd[0];
    sigma[0] = sigmaInd[0];
    for (int i = 0; i < ali->nSites; i++) mu[i + 2] = muInd[i + 1];
    for (int i = 0; i < ali->nSites; i++) sigma[i + 2] = sigmaInd[i + 1];
    int shift = 2 + ali->nSites + ali->nSites * (ali->nSites - 1) / 2;
    for (int i = 0; i < ali->nSites * ali->nCodes; i++)
        mu[shift + i] = muInd[1 + ali->nSites + i];
    for (int i = 0; i < ali->nSites * ali->nCodes; i++)
        sigma[shift + i] = sigmaInd[1 + ali->nSites + i];

    /* Stochastically optimize KL(Q||P(params|data)) for Gaussian Q */
    numeric_t ELBO =
        EstimateGaussianVariationalApproximation(VBayesPairHierarchicalNonCentGibbs,
        data, mu, sigma, n, options->vSamples, eps, options->maxIter, crit);

    /* Copy means and variances into full parameter block */
    for (int i = 0; i < n; i++) x[i] = mu[i];
    for (int i = 0; i < n; i++) x[i + n] = sigma[i];

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test set of parameters */
    // fprintf(stderr, "%d sites x %d states", ali->nSites, ali->nCodes);
    // for (int i = 0; i < ali->nSites; i++)
    //     for (int ai = 0; ai < ali->nCodes; ai++)
    //         xHi(i, ai) = (numeric_t) ai;
    // for (int i = 0; i < ali->nSites-1; i++)
    //     for (int j = i + 1; j < ali->nSites; j++)
    //         for (int ai = 0; ai < ali->nCodes; ai++)
    //             for (int aj = 0; aj < ali->nCodes; aj++)
    //                 xEij(i, j, ai, aj) = (numeric_t) (ai + aj);
    /* --------------------------------^DEBUG^--------------------------------*/
}

numeric_t VBayesPairHierarchicalNonCentGibbs(void *data, const numeric_t *xB, 
    numeric_t *gB, const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a pairwise hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];

    /* Initialize -LogPosterior & gradient to zero */
    numeric_t negLogP = 0;
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Global hyperparameters */
    const numeric_t scaleH = exp(xB[0]);
    const numeric_t scaleE = exp(xB[1]);
    numeric_t *gLogScaleH = &(gB[0]);
    numeric_t *gLogScaleE = &(gB[1]);

    /* Local hyperparameters */
    const numeric_t *lambdas = &(xB[2]);
    numeric_t *gLambdas = &(gB[2]);

    /* Non-centered parameters */
    int offset = 2 + ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Gradient: sitewise marginals of the data */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            dHi(i, ai) = -ali->nEff * fi(i, ai);

    /* Gradient: pairwise marginals of the data */
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    dEij(i, j, ai, aj) = -ali->nEff * fij(i, j, ai, aj);

    /* Gradient: marginals of the model by parallel Gibbs samplers */
    int gSweeps = options->gSweeps;
    int gChains = options->gChains;
    letter_t *sample = (letter_t *) malloc(gChains * gSweeps * ali->nSites
        * sizeof(letter_t));

    /* Presample from the pseudo-random number generator for thread safety */
    int nSteps = gChains * gSweeps * ali->nSites;
    int *siteI = (int *) malloc(nSteps * sizeof(int));
    double *codeU = (double *) malloc(nSteps * sizeof(double));
    for (int i = 0; i < nSteps; i++) siteI[i] = genrand_int31() % ali->nSites;
    for (int i = 0; i < nSteps; i++) codeU[i] = genrand_real3();

    /* Parallelize across the chains */
    #pragma omp parallel for
    for (int c = 0; c < gChains; c++) {
        /* Samples gSweeps sequences in each chain */
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        for (int s = 0; s < gSweeps; s++) {
            /* Sweep nSites positions */
            for (int sx = 0; sx < ali->nSites; sx++) {
                /* Pick a random site */
                int i = siteI[c * gSweeps * ali->nSites + s * ali->nSites + sx];

                /* Compute conditional CDF at the site */
                for (int a = 0; a < ali->nCodes; a++)
                    P[a] = exp(lambdaHi(i)) * xHi(i, a);
                for (int j = 0; j < i; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        P[a] += exp(lambdaEij(i, j))
                             * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
                for (int j = i + 1; j < ali->nSites; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        P[a] += exp(lambdaEij(i, j))
                             * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
                numeric_t scale = P[0];
                for (int a = 1; a < ali->nCodes; a++)
                    scale = (scale >= P[a] ? scale : P[a]);
                for (int a = 0; a < ali->nCodes; a++) P[a] = exp(P[a] - scale);
                for (int a = 1; a < ali->nCodes; a++) P[a] += P[a - 1];

                /* Choose a new code for the site */
                double u = P[ali->nCodes - 1] *
                    codeU[c * gSweeps * ali->nSites + s * ali->nSites + sx];
                int aNew = 0;
                while (u > P[aNew]) aNew++;
                ali->samples[c * ali->nSites + i] = aNew;
            }

            /* Copy sequence into the global sample */
            for (int i = 0; i < ali->nSites; i++)
                sample[c * gSweeps * ali->nSites + s * ali->nSites + i] =
                    ali->samples[c * ali->nSites + i];
        }
        free(P);
    }

    /* Contribute to global gradient for centered parameters */
    numeric_t nRatio =
        ((numeric_t) ali->nEff) / ((numeric_t) (gChains * gSweeps));
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++)
        for (int c = 0; c < gChains; c++)
            for (int s = 0; s < gSweeps; s++)
                dHi(i, sample[c * gSweeps * ali->nSites + s * ali->nSites + i])
                     += nRatio;
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int c = 0; c < gChains; c++)
                for (int s = 0; s < gSweeps; s++)
                    dEij(i, j, 
                        sample[c * gSweeps * ali->nSites + s * ali->nSites + i],
                        sample[c * gSweeps * ali->nSites + s * ali->nSites + j])
                        += nRatio;

    free(sample);
    free(siteI);
    free(codeU);

    /* Transform gradients for non-centered parameterization */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            gLambdaHi(i) += exp(lambdaHi(i)) * xHi(i, ai) * dHi(i, ai);
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    gLambdaEij(i,j) += exp(lambdaEij(i, j)) * xEij(i, j, ai, aj)
                                                            * dEij(i, j, ai, aj);
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            dHi(i, ai) *= exp(lambdaHi(i));
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    dEij(i, j, ai, aj) *= exp(lambdaEij(i, j));


    /* Estimate the log partition function by Annealed Importance Sampling */
    numeric_t logZ = 0;
    /* Initial distribution: fields only */
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t Zi = 0;
        for (int ai = 0; ai < ali->nCodes; ai++)
            Zi += exp(exp(lambdaHi(i)) * xHi(i, ai));
        logZ += log(Zi);
    }

    /* Draw an initial sequence from the starting distribution */
    numeric_t H = 0;
    letter_t *S = (letter_t *) malloc(ali->nSites * sizeof(letter_t));
    numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
    for (int i = 0; i < ali->nSites; i++) {
        /* Compute conditional CDF at the site */
        for (int a = 0; a < ali->nCodes; a++)
            P[a] = exp(lambdaHi(i)) * xHi(i, a);
        numeric_t scale = P[0];
        for (int a = 1; a < ali->nCodes; a++)
            scale = (scale >= P[a] ? scale : P[a]);
        for (int a = 0; a < ali->nCodes; a++) P[a] = exp(P[a] - scale);
        for (int a = 1; a < ali->nCodes; a++) P[a] += P[a - 1];

        /* Choose a new code for the site */
        double u = genrand_real3() * P[ali->nCodes - 1];
        int aNew = 0;
        while (u > P[aNew]) aNew++;
        S[i] = aNew;
        H += exp(lambdaHi(i)) * xHi(i, S[i]);
    }

    /* Annealed Sampling */
    int nAIS = 20 * ali->nSites;
    for (int bx = 0; bx < nAIS; bx++) {
        /* Pick a random site */
        int i = genrand_int31() % ali->nSites;

        /* Compute conditional CDF at the site at temperature BETA */
        numeric_t beta = ((numeric_t) bx) / ((numeric_t) (nAIS - 1));
        for (int a = 0; a < ali->nCodes; a++)
            P[a] = exp(lambdaHi(i)) * xHi(i, a);
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                P[a] += beta * exp(lambdaEij(i, j))
                        * xEij(i, j, a, S[j]);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                P[a] += beta * exp(lambdaEij(i, j))
                        * xEij(i, j, a, S[j]);

        numeric_t scale = P[0];
        for (int a = 1; a < ali->nCodes; a++)
            scale = (scale >= P[a] ? scale : P[a]);
        for (int a = 0; a < ali->nCodes; a++) P[a] = exp(P[a] - scale);
        for (int a = 1; a < ali->nCodes; a++) P[a] += P[a - 1];

        /* Choose a new code for the site */
        double u = genrand_real3() * P[ali->nCodes - 1];
        int aNew = 0;
        while (u > P[aNew]) aNew++;
        S[i] = aNew;

        /* Contribute changes in Hamiltonian to partition function */
        if (bx % ali->nSites == 0) {
            numeric_t deltaB = ((numeric_t) ali->nSites)
                               / ((numeric_t) (nAIS - 1));
            for (int i = 0; i < ali->nSites - 1; i++)
                for (int j = i + 1; j < ali->nSites; j++)
                    logZ += deltaB * exp(lambdaEij(i, j))
                            * xEij(i, j, S[i], S[j]);
        }
    }
    free(S);
    free(P);

    /* Compute the negative log likelihood using estimated Log[Z] */
    #pragma omp parallel for reduction(+:negLogP)
    for (int s = 0; s < ali->nSeqs; s++) {
        numeric_t H = 0;
        for (int i = 0; i < ali->nSites; i++)
            H += exp(lambdaHi(i)) * xHi(i, seq(s, i));
        for (int i = 0; i < ali->nSites - 1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                H += exp(lambdaEij(i, j)) * xEij(i, j, seq(s, i), seq(s, j));
        negLogP +=  -ali->weights[s] * H;
    }
    negLogP += ali->nEff * logZ;

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Output estimated logZ */
    // assert(isfinite(logZ));
    /* --------------------------------^DEBUG^--------------------------------*/

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Zero contributions from likelihood */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    /* --------------------------------^DEBUG^--------------------------------*/

    /* Half-Cauchy(1) prior over the global scale parameters */
    negLogP += -log(scaleH) + log(1.0 + scaleH * scaleH) - log(2.0 / PI);
    negLogP += -log(scaleE) + log(1.0 + scaleE * scaleE) - log(2.0 / PI);
    gLogScaleH[0] = -1.0 + 2.0 * scaleH * scaleH / (1.0 + scaleH * scaleH);
    gLogScaleE[0] = -1.0 + 2.0 * scaleE * scaleE / (1.0 + scaleE * scaleE);

    /* Prior over the variances */
    if (options->hyperprior == PRIOR_HALFCAUCHYPLUS) {
        /* The Horseshoe+: Double Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            numeric_t sigmaHi = exp(lambdaHi(i));
            if (abs(scaleH - sigmaHi) < 1E-3) {
                /* Numerically stable computation as sigma -> scale */
                negLogP += -log(2.0 / (PI * PI));
            } else {
                /* Double indicator functions negate negative logs */
                numeric_t I = (numeric_t) (2 * (scaleH > sigmaHi) - 1);
                negLogP += -log(4.0 / (PI * PI)) - log(scaleH) - lambdaHi(i)
                           - log(I * (log(scaleH) - lambdaHi(i)))
                           + log(I * (scaleH * scaleH - exp(2 * lambdaHi(i))));
                gLambdaHi(i) += - 1.0
                                + 1.0 / (log(scaleH) - lambdaHi(i))
                                - 2.0 * exp(2 * lambdaHi(i))
                                  / (scaleH * scaleH - exp(2 * lambdaHi(i)));
                gLogScaleH[0] += - 1.0
                                 - 1.0 / (log(scaleH) - lambdaHi(i))
                                 + 2.0 * scaleH * scaleH
                                    / (scaleH * scaleH - exp(2 * lambdaHi(i)));
            }
        }
        for (int i = 0; i < ali->nSites - 1; i++) 
            for (int j = i + 1; j < ali->nSites; j++) {
                numeric_t sigmaEij = exp(lambdaEij(i, j));
                if (abs(scaleE - sigmaEij) < 1E-3) {
                    /* Numerically stable computation as sigma -> scale */
                    negLogP += -log(2.0 / (PI * PI));
                } else {
                    /* Double indicator functions negate negative logs */
                    numeric_t I = (numeric_t) (2 * (scaleE > sigmaEij) - 1);
                    negLogP += -log(4.0 / (PI * PI)) - log(scaleE) - lambdaEij(i, j)
                               - log(I * (log(scaleE) - lambdaEij(i, j)))
                               + log(I * (scaleE * scaleE - exp(2 * lambdaEij(i, j))));
                    gLambdaEij(i,j) += - 1.0
                                    + 1.0 / (log(scaleE) - lambdaEij(i, j))
                                    - 2.0 * exp(2 * lambdaEij(i, j))
                                      / (scaleE * scaleE - exp(2 * lambdaEij(i, j)));
                    gLogScaleE[0] += - 1.0
                                     - 1.0 / (log(scaleE) - lambdaEij(i, j))
                                     + 2.0 * scaleE * scaleE
                                        / (scaleE * scaleE - exp(2 * lambdaEij(i, j)));
                }
            }
    } else if (options->hyperprior == PRIOR_EXPONENTIAL) {
        /* Laplacian prior: exponentially distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            negLogP += -log(2.0) - log(scaleH) - 2.0 * lambdaHi(i)
                       + scaleH * exp(2.0 * lambdaHi(i));
            gLambdaHi(i) += -2.0 + 2.0 * scaleH * exp(2.0 * lambdaHi(i));
            gLogScaleH[0] += -1.0 + scaleH * exp(2.0 * lambdaHi(i));
        }
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                negLogP += -log(2.0) - log(scaleE) - 2.0 * lambdaEij(i,j)
                       + scaleE * exp(2.0 * lambdaEij(i,j));
                gLambdaEij(i,j) += -2.0
                                + 2.0 * scaleE * exp(2.0 * lambdaEij(i,j));
                gLogScaleE[0] += -1.0 + scaleE * exp(2.0 * lambdaEij(i,j));
            }
    } else {
        /* The Horseshoe: Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            negLogP += -lambdaHi(i) + log(scaleH * scaleH + exp(2 * lambdaHi(i)))
                       -log(2 * scaleH / PI);
            gLambdaHi(i) += -1 + 2.0 * exp(2 * lambdaHi(i))
                      / (scaleH * scaleH + exp(2 * lambdaHi(i)));
            gLogScaleH[0] += -1.0 + 2.0 * scaleH * scaleH
                / (scaleH * scaleH + exp(2 * lambdaHi(i)));
        }
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                negLogP += -lambdaEij(i, j)
                           + log(scaleE * scaleE + exp(2 * lambdaEij(i, j)))
                           -log(2 * scaleE / PI);
                gLambdaEij(i,j) += -1 + 2.0 * exp(2 * lambdaEij(i, j))
                      / (scaleE * scaleE + exp(2 * lambdaEij(i, j)));
                gLogScaleE[0] += -1.0 + 2.0 * scaleE * scaleE
                / (scaleE * scaleE + exp(2 * lambdaEij(i, j)));
            }
    }

    /* Standard Normal prior over h & e */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) {
            dHi(i, ai) += xHi(i, ai);
            negLogP += 0.5 * xHi(i, ai) * xHi(i, ai) + 0.5 * log(2 * PI);
        }
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++) {
                    dEij(i, j, ai, aj) += xEij(i, j, ai, aj);
                    negLogP += 0.5 * xEij(i, j, ai, aj) * xEij(i, j, ai, aj)
                             + 0.5 * log(2 * PI);
                }

    return negLogP;
}

numeric_t VBayesPairHierarchicalNonCentPL(void *data, const numeric_t *xB, numeric_t *gB, 
    const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a pairwise hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];

    /* Initialize -LogPosterior & gradient to zero */
    numeric_t negLogP = 0;
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Global hyperparameters */
    const numeric_t scaleH = exp(xB[0]);
    const numeric_t scaleE = exp(xB[1]);
    numeric_t *gLogScaleH = &(gB[0]);
    numeric_t *gLogScaleE = &(gB[1]);

    /* Local hyperparameters */
    const numeric_t *lambdas = &(xB[2]);
    numeric_t *gLambdas = &(gB[2]);

    /* Non-centered parameters */
    int offset = 2 + ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Negative log-pseudolikelihood */
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        /* Gradients of logSigma at the site */
        numeric_t *gSiteLambda =
            (numeric_t *) malloc(ali->nSites * sizeof(numeric_t));
        for (int j = 0; j < ali->nSites; j++) gSiteLambda[j] = 0;

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = exp(lambdaEij(i, j)) * xEij(i, j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = exp(lambdaEij(i, j)) * xEij(i, j, a, b);
        for (int a = 0; a < ali->nCodes; a++)
            siteH(i, a) = exp(lambdaHi(i)) * xHi(i, a);

        numeric_t *Di = (numeric_t *) malloc(ali->nCodes * ali->nCodes
        * ali->nSites * sizeof(numeric_t));
        for (int d = 0; d < ali->nCodes * ali->nCodes * ali->nSites; d++)
            Di[d] = 0.0;

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Compute potentials */
            for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));

            /* Conditional distribution given sequence background */
            numeric_t scale = H[0];
            for (int a = 1; a < ali->nCodes; a++)
                scale = (scale >= H[a] ? scale : H[a]);
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


            /* Log-likelihood contributions are scaled by sequence weight */
            numeric_t w = ali->weights[s];  
            siteFx -= w * log(P[seq(s, i)]);

            /* Non-centered reparameterization introduces factor of sigma */
            /* Field gradient */
            siteDH(i, seq(s, i)) -= w * exp(lambdaHi(i));
            for (int a = 0; a < ali->nCodes; a++)
                siteDH(i, a) -= -w * P[a] * exp(lambdaHi(i));

            /* Couplings gradient */
            int ix = seq(s, i);
            for (int j = 0; j < i; j++)
                siteDE(j, ix, seq(s, j)) -= w * exp(lambdaEij(i,j));
            for (int j = i + 1; j < ali->nSites; j++)
                siteDE(j, ix, seq(s, j)) -= w * exp(lambdaEij(i,j));
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a] * exp(lambdaEij(i,j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a] * exp(lambdaEij(i,j));

            /* LogSigma gradients */
            numeric_t gradHSum = 0;
            for (int a = 0; a < ali->nCodes; a++) gradHSum += P[a] * siteH(i, a);
            gSiteLambda[i] += exp(lambdaHi(i)) * (gradHSum - siteH(i, seq(s, i)));

            for (int j = 0; j < i; j++) {
                numeric_t gradESum = 0;
                for (int a = 0; a < ali->nCodes; a++)
                    gradESum += P[a] * siteE(j, a, seq(s, j));
                gSiteLambda[j] += exp(lambdaEij(i, j))
                                 * (gradESum - siteE(j, seq(s, i), seq(s, j)));
            }
            for (int j = i + 1; j < ali->nSites; j++) {
                numeric_t gradESum = 0;
                for (int a = 0; a < ali->nCodes; a++)
                    gradESum += P[a] * siteE(j, a, seq(s, j));
                gSiteLambda[j] += exp(lambdaEij(i, j))
                                 * (gradESum - siteE(j, seq(s, i), seq(s, j)));
            }
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        negLogP += siteFx;
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int a = 0; a < ali->nCodes; a++) dHi(i, a) += siteDH(i, a);

        /* Gradients over log sigma */
        gLambdaHi(i) += gSiteLambda[i];
        for (int j = 0; j < i; j++)
            gLambdaEij(i,j) += gSiteLambda[j];
        for (int j = i + 1; j < ali->nSites; j++)
            gLambdaEij(i,j) += gSiteLambda[j];

        free(Xi);
        free(Di);
        free(gSiteLambda);
        }

        free(H);
        free(P);
    }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Zero contributions from likelihood */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    /* --------------------------------^DEBUG^--------------------------------*/

    /* Half-Cauchy(1) prior over the global scale parameters */
    negLogP += -log(scaleH) + log(1.0 + scaleH * scaleH) - log(2.0 / PI);
    negLogP += -log(scaleE) + log(1.0 + scaleE * scaleE) - log(2.0 / PI);
    gLogScaleH[0] = -1.0 + 2.0 * scaleH * scaleH / (1.0 + scaleH * scaleH);
    gLogScaleE[0] = -1.0 + 2.0 * scaleE * scaleE / (1.0 + scaleE * scaleE);

    /* Prior over the variances */
    if (options->hyperprior == PRIOR_HALFCAUCHYPLUS) {
        /* The Horseshoe+: Double Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            numeric_t sigmaHi = exp(lambdaHi(i));
            if (abs(scaleH - sigmaHi) < 1E-3) {
                /* Numerically stable computation as sigma -> scale */
                negLogP += -log(2.0 / (PI * PI));
            } else {
                /* Double indicator functions negate negative logs */
                numeric_t I = (numeric_t) (2 * (scaleH > sigmaHi) - 1);
                negLogP += -log(4.0 / (PI * PI)) - log(scaleH) - lambdaHi(i)
                           - log(I * (log(scaleH) - lambdaHi(i)))
                           + log(I * (scaleH * scaleH - exp(2 * lambdaHi(i))));
                gLambdaHi(i) += - 1.0
                                + 1.0 / (log(scaleH) - lambdaHi(i))
                                - 2.0 * exp(2 * lambdaHi(i))
                                  / (scaleH * scaleH - exp(2 * lambdaHi(i)));
                gLogScaleH[0] += - 1.0
                                 - 1.0 / (log(scaleH) - lambdaHi(i))
                                 + 2.0 * scaleH * scaleH
                                    / (scaleH * scaleH - exp(2 * lambdaHi(i)));
            }
        }
        for (int i = 0; i < ali->nSites; i++) 
            for (int j = i + 1; j < ali->nSites; j++) {
                numeric_t sigmaEij = exp(lambdaEij(i, j));
                if (abs(scaleE - sigmaEij) < 1E-3) {
                    /* Numerically stable computation as sigma -> scale */
                    negLogP += -log(2.0 / (PI * PI));
                } else {
                    /* Double indicator functions negate negative logs */
                    numeric_t I = (numeric_t) (2 * (scaleE > sigmaEij) - 1);
                    negLogP += -log(4.0 / (PI * PI)) - log(scaleE) - lambdaEij(i, j)
                               - log(I * (log(scaleE) - lambdaEij(i, j)))
                               + log(I * (scaleE * scaleE - exp(2 * lambdaEij(i, j))));
                    gLambdaEij(i,j) += - 1.0
                                    + 1.0 / (log(scaleE) - lambdaEij(i, j))
                                    - 2.0 * exp(2 * lambdaEij(i, j))
                                      / (scaleE * scaleE - exp(2 * lambdaEij(i, j)));
                    gLogScaleE[0] += - 1.0
                                     - 1.0 / (log(scaleE) - lambdaEij(i, j))
                                     + 2.0 * scaleE * scaleE
                                        / (scaleE * scaleE - exp(2 * lambdaEij(i, j)));
                }
            }
    } else {
        /* The Horseshoe: Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            negLogP += -lambdaHi(i) + log(scaleH * scaleH + exp(2 * lambdaHi(i)))
                       -log(2 * scaleH / PI);
            gLambdaHi(i) += -1 + 2.0 * exp(2 * lambdaHi(i))
                      / (scaleH * scaleH + exp(2 * lambdaHi(i)));
            gLogScaleH[0] += -1.0 + 2.0 * scaleH * scaleH
                / (scaleH * scaleH + exp(2 * lambdaHi(i)));
        }
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                negLogP += -lambdaEij(i, j)
                           + log(scaleE * scaleE + exp(2 * lambdaEij(i, j)))
                           -log(2 * scaleE / PI);
                gLambdaEij(i,j) += -1 + 2.0 * exp(2 * lambdaEij(i, j))
                      / (scaleE * scaleE + exp(2 * lambdaEij(i, j)));
                gLogScaleE[0] += -1.0 + 2.0 * scaleE * scaleE
                / (scaleE * scaleE + exp(2 * lambdaEij(i, j)));
            }
    }

    /* Standard Normal prior over h & e */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) {
            dHi(i, ai) += xHi(i, ai);
            negLogP += 0.5 * xHi(i, ai) * xHi(i, ai);
        }
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++) {
                    dEij(i, j, ai, aj) += xEij(i, j, ai, aj);
                    negLogP += 0.5 * xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
                }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate normal with shifting variances */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    // for (int i = 0; i < n; i++) {
    //     numeric_t mu = (numeric_t) (i % 4 + 1);
    //     numeric_t sigma = (numeric_t) (i + 1);
    //     sigma = 3.0;
    //     negLogP += (xB[i] - mu) * (xB[i] - mu) / (2.0 * sigma * sigma);
    //     gB[i] = (xB[i] - mu) / (sigma * sigma);
    // }
    /* --------------------------------^DEBUG^--------------------------------*/

    return negLogP;
}

numeric_t VBayesSiteHierarchicalNonCent(void *data, const numeric_t *xB, numeric_t *gB, 
    const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a pairwise hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];

    /* Initialize -LogPosterior & gradient to zero */
    numeric_t negLogP = 0;
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Global hyperparameters */
    const numeric_t scaleH = exp(xB[0]);
    numeric_t *gLogScaleH = &(gB[0]);

    /* Local hyperparameters */
    const numeric_t *lambdas = &(xB[1]);
    numeric_t *gLambdas = &(gB[1]);

    /* Non-centered parameters */
    int offset = 1 + ali->nSites;
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Negative log likelihood */
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t siteFx = 0.0;

        /* Compute site-independent distribution at i */
        for (int a = 0; a < ali->nCodes; a++)
            H[a] = exp(lambdaHi(i)) * xHi(i, a);
        numeric_t scale = H[0];
        for (int a = 1; a < ali->nCodes; a++)
            scale = (scale >= H[a] ? scale : H[a]);
        for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
        numeric_t Z = 0;
        for (int a = 0; a < ali->nCodes; a++) Z += P[a];
        numeric_t Zinv = 1.0 / Z;
        for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;

        /* Site negative log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++)
            negLogP -= ali->weights[s] * log(P[seq(s, i)]);

        for (int a = 0; a < ali->nCodes; a++)
            dHi(i, a) = ali->nEff * (P[a] - fi(i, a));

        free(H);
        free(P);
    }

    /* Transform gradients for non-centered parameterization */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            gLambdaHi(i) += exp(lambdaHi(i)) * xHi(i, ai) * dHi(i, ai);
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            dHi(i, ai) *= exp(lambdaHi(i));

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Zero contributions from likelihood */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    /* --------------------------------^DEBUG^--------------------------------*/

    /* Half-Cauchy(1) prior over the global scale parameters */
    negLogP += -log(scaleH) + log(1.0 + scaleH * scaleH) - log(2.0 / PI);
    gLogScaleH[0] = -1.0 + 2.0 * scaleH * scaleH / (1.0 + scaleH * scaleH);
    
    /* Prior over the variances */
    if (options->hyperprior == PRIOR_HALFCAUCHYPLUS) {
        /* The Horseshoe+: Double Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            numeric_t sigmaHi = exp(lambdaHi(i));
            if (abs(scaleH - sigmaHi) < 1E-3) {
                /* Numerically stable computation as sigma -> scale */
                negLogP += -log(2.0 / (PI * PI));
            } else {
                /* Double indicator functions negate negative logs */
                numeric_t I = (numeric_t) (2 * (scaleH > sigmaHi) - 1);
                negLogP += -log(4.0 / (PI * PI)) - log(scaleH) - lambdaHi(i)
                           - log(I * (log(scaleH) - lambdaHi(i)))
                           + log(I * (scaleH * scaleH - exp(2 * lambdaHi(i))));
                gLambdaHi(i) += - 1.0
                                + 1.0 / (log(scaleH) - lambdaHi(i))
                                - 2.0 * exp(2 * lambdaHi(i))
                                  / (scaleH * scaleH - exp(2 * lambdaHi(i)));
                gLogScaleH[0] += - 1.0
                                 - 1.0 / (log(scaleH) - lambdaHi(i))
                                 + 2.0 * scaleH * scaleH
                                    / (scaleH * scaleH - exp(2 * lambdaHi(i)));
            }
        }
    } else if (options->hyperprior == PRIOR_EXPONENTIAL) {
        /* Laplacian prior: exponentially distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            negLogP += -log(2.0) - log(scaleH) - 2.0 * lambdaHi(i)
                       + scaleH * exp(2.0 * lambdaHi(i));
            gLambdaHi(i) += -2.0 + 2.0 * scaleH * exp(2.0 * lambdaHi(i));
            gLogScaleH[0] += -1.0 + scaleH * exp(2.0 * lambdaHi(i));
        }
    } else {
        /* The Horseshoe: Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            negLogP += -lambdaHi(i) + log(scaleH * scaleH + exp(2 * lambdaHi(i)))
                       -log(2 * scaleH / PI);
            gLambdaHi(i) += -1 + 2.0 * exp(2 * lambdaHi(i))
                      / (scaleH * scaleH + exp(2 * lambdaHi(i)));
            gLogScaleH[0] += -1.0 + 2.0 * scaleH * scaleH
                / (scaleH * scaleH + exp(2 * lambdaHi(i)));
        }
    }

    /* Standard Normal prior over h & e */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) {
            dHi(i, ai) += xHi(i, ai);
            negLogP += 0.5 * xHi(i, ai) * xHi(i, ai);
        }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate normal with shifting variances */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    // for (int i = 0; i < n; i++) {
    //     numeric_t mu = (numeric_t) (i % 4 + 1);
    //     numeric_t sigma = (numeric_t) (i + 1);
    //     sigma = 3.0;
    //     negLogP += (xB[i] - mu) * (xB[i] - mu) / (2.0 * sigma * sigma);
    //     gB[i] = (xB[i] - mu) / (sigma * sigma);
    // }
    /* --------------------------------^DEBUG^--------------------------------*/

    return negLogP;
}

numeric_t VBayesPairHierarchical(void *data, const numeric_t *xB, numeric_t *gB, 
    const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a pairwise hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    // options_t *options = (options_t *) d[1];

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate normal with shifting variances */
    // for (int i = 0; i < n; i++) {
    //     numeric_t mu = 3 * (numeric_t) (i + 1);
    //     numeric_t sigma = (numeric_t) (i + 1);
    //     negLogP += (xB[i] - mu) * (xB[i] - mu) / (2.0 * sigma * sigma);
    //     gB[i] = (xB[i] - mu) / (sigma * sigma);
    // }
    /* --------------------------------^DEBUG^--------------------------------*/

    /* Initialize gradient to zero */
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Hyperparameters are indexed before parameter set */
    // const numeric_t *lambdas = xB;
    // numeric_t *gLambdas = gB;
    int offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    /* Parameter set is notated */
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Negative log-pseudolikelihood */
    numeric_t negLogP = 0;
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int a = 0; a < ali->nCodes; a++) siteH(i, a) = xHi(i, a);

        numeric_t *Di = (numeric_t *) malloc(ali->nCodes * ali->nCodes
        * ali->nSites * sizeof(numeric_t));
        for (int d = 0; d < ali->nCodes * ali->nCodes * ali->nSites; d++)
            Di[d] = 0.0;

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Compute potentials */
            for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));

            /* Conditional distribution given sequence background */
            numeric_t scale = H[0];
            for (int a = 1; a < ali->nCodes; a++)
                scale = (scale >= H[a] ? scale : H[a]);
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


            /* Log-likelihood contributions are scaled by sequence weight */
            numeric_t w = ali->weights[s];  
            siteFx -= w * log(P[seq(s, i)]);

            /* Field gradient */
            siteDH(i, seq(s, i)) -= w;
            for (int a = 0; a < ali->nCodes; a++)
                siteDH(i, a) -= -w * P[a];

            /* Couplings gradient */
            int ix = seq(s, i);
            for (int j = 0; j < i; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = i + 1; j < ali->nSites; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        negLogP += siteFx;
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int a = 0; a < ali->nCodes; a++) dHi(i, a) += siteDH(i, a);
        free(Xi);
        free(Di);
        }

        free(H);
        free(P);
    }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate standard normal */
    // negLogP = 0;
    // for (int i = 0; i < n; i++) gB[i] = 0;
    numeric_t sigma = 10.0;
    for (int i = 0; i < n; i++)
        negLogP += xB[i] * xB[i] / (2.0 * sigma * sigma);
    for (int i = 0; i < n; i++)
        gB[i] += xB[i] / (sigma * sigma);
    /* ------------------------G-------^DEBUG^--------------------------------*/


    /* Gaussian priors */
    // for (int i = 0; i < ali->nSites; i++)
    //     for (int ai = 0; ai < ali->nCodes; ai++) {
    //         dHi(i, ai) += lambdaHi(i) * 2.0 * xHi(i, ai);
    //         negLogP += lambdaHi(i) * xHi(i, ai) * xHi(i, ai);
    //     }

    // for (int i = 0; i < ali->nSites-1; i++)
    //     for (int j = i + 1; j < ali->nSites; j++)
    //         for (int ai = 0; ai < ali->nCodes; ai++)
    //             for (int aj = 0; aj < ali->nCodes; aj++) {
    //                 dEij(i, j, ai, aj) += lambdaEij(i, j)
    //                     * 2.0 * xEij(i, j, ai, aj);
    //                 negLogP += lambdaEij(i, j)
    //                     * xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
    //             }
    // return negLogP;

    return negLogP;
}

void EstimateSiteLambdasBayes(numeric_t *lambdas, alignment_t *ali,
    options_t *options) {
    /* Estimate site hyperparameters lambda_hi = 1 / (2 * sigma_hi^2) by HMC */
    /* Each site is modeled as a Boltzmann-like distribution where the weights 
       for each state are drawn from a Gaussian of unknown variance */

    fprintf(stderr, "Estimating site hyperparameters lh = 1/2 inverse variance\n");
    fprintf(stderr, "site\t<lh>\t95%% credible\teps\taccept rate\n");
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *C = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        for (int ai = 0; ai < ali->nCodes; ai++) C[ai] = fi(i, ai) * ali->nEff;

        /* ------------------------------_DEBUG_----------------------------- */
        /* Rescale the counts */
        // for (int ai = 0; ai < ali->nCodes; ai++) C[ai] = (C[ai] + 10) * 100.0;
        // for (int ai = 0; ai < ali->nCodes; ai++) C[ai] *= 100.0;
        // for (int ai = 0; ai < ali->nCodes; ai++) C[ai] /= 10.0;
        // for (int ai = 0; ai < ali->nCodes; ai++) C[ai] *= 100.0 / ali->nEff;
        /* ------------------------------^DEBUG^----------------------------- */

        /* Array of void pointers provides relevant parameters */
        numeric_t scale = 2;
        numeric_t counts = 0;
        for (int ai = 0; ai < ali->nCodes; ai++) counts += C[ai];
        void *data[3] = {(void *)C, (void *)&scale, (void *)&counts};

        /* Tuned parameters for HMC */
        int n = ali->nCodes + 1;        /* Number of variables */
        int s = 1E3;                    /* Number of samples to collect */
        int L = 100;                    /* Integration number of steps */
        numeric_t eps = 0.01;           /* Median step size +/- 10-fold */
        int warmup = 1E3;               /* Number of samples to discard */

        /* Samples from HMC */
        numeric_t *X = (numeric_t *) malloc(n * s * sizeof(numeric_t));
        for (int ix = 0; ix < n * s; ix++) X[ix] = 0;
        numeric_t accRate =
            SampleHamiltonianMonteCarlo(HMCSiteHNonCenter, HMCSiteHGradNonCenter,
            data, X, n, s, L, &eps, warmup);

        /* Estimate posterior average from HMC */
        numeric_t invS = 1.0 / (numeric_t) s;
        numeric_t sigmaAv = 0;
        for (int sx = 0; sx < s; sx++) sigmaAv += invS * exp(X[sx * n]);

        /* Compute statistics of lambda = 1/2 the inverse variance */
        numeric_t *lambdah = (numeric_t *) malloc(s * sizeof(numeric_t));
        for (int sx = 0; sx < s; sx++)
            lambdah[sx] = 1.0 / (2.0 * exp(2.0 * X[sx * n]));
        numeric_t lambdaAv = 0;
        numeric_t lambdaVar = 0;
        for (int sx = 0; sx < s; sx++) lambdaAv += invS * lambdah[sx];
        for (int sx = 0; sx < s; sx++)
            lambdaVar += invS * pow(lambdah[sx] - lambdaAv, 2);

        /* Find the 2.5% and 97.5% quantiles for the 95% credible interval */
        numeric_t lowLambda = QuickSelect(lambdah, s, (int) floor(0.025 * s));
        for (int sx = 0; sx < s; sx++)
            lambdah[sx] = 1.0 / (2.0 * exp(2.0 * X[sx * n]));
        numeric_t highLambda = QuickSelect(lambdah, s, (int) floor(0.975 * s));
        free(lambdah);

        fprintf(stderr, "%d\t%3.2f\t[%3.2f, %3.2f]\t%5.4f\t%3.1f\n", i + 1, 
            lambdaAv, lowLambda, highLambda, eps, 100 * accRate);

        /* Set the hyperparameters with the estimated posterior mean */
        lambdaHi(i) = lambdaAv;

        /* ------------------------------_DEBUG_----------------------------- */
        /* Compute and print posterior means */
        // for (int ix = 1; ix < n; ix++) {
        //     numeric_t avH = 0;
        //     for (int sx = 0; sx < s; sx++) 
        //         avH += exp(X[sx * n]) * X[sx * n + ix];
        //     avH /= (numeric_t) s;
        //     fprintf(stdout, "%f\t", avH);
        // }
        // fprintf(stdout, "\n");
        // if (i == ali->nSites - 1) exit(0);
        /* ------------------------------^DEBUG^----------------------------- */

        /* ------------------------------_DEBUG_----------------------------- */
        /* Dump HMC samples for analysis */
        // if (i == 42) {
        //     for (int sx = 0; sx < s; sx++) {
        //         for (int ix = 0; ix < n; ix++)
        //             fprintf(stdout, "%f\t", X[sx * n + ix]);
        //         fprintf(stdout, "\n");
        //     }
        //     exit(0);
        // }
        /* ------------------------------^DEBUG^----------------------------- */
        free(X);
        free(C);
    }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Print marginals to console */
    // for (int i = 0; i < ali->nSites; i++) {
    //     if (i%10 == 0) {
    //        for (int ai = 0; ai < ali->nCodes; ai++)
    //         fprintf(stderr, "\033[0;30;47m  %c   \x1B[0m",
    //             ali->alphabet[ai +
    //                 (options->estimatorMAP == INFER_MAP_PLM_GAPREDUCE)]);
    //         fprintf(stderr, "\n"); 
    //     }
    //     for (int ai = 0; ai < ali->nCodes; ai++)
    //         if (fi(i,ai) > 0.5) {
    //             fprintf(stderr, "\033[0;30;47m%5.1f\x1B[0m ", 100 * fi(i,ai));
    //         } else if (fi(i,ai) > 1.0 / ((numeric_t) ali->nCodes)) {
    //             fprintf(stderr, "%5.1f ", 100 * fi(i,ai));
    //         } else {
    //             fprintf(stderr, "      ");
    //         }
    //     fprintf(stderr, "\t%4.3f\n", lambdaHi(i));
    // }
    // exit(0);
    /* --------------------------------^DEBUG^--------------------------------*/
}

numeric_t HMCSiteH(void *data, const numeric_t *x, const int n) {
    /* Computes the (unnormalized) negative log posterior of single-site 
       hierarchical model */
    numeric_t f = 0;

    void **d = (void **)data;
    numeric_t *C = (numeric_t *) d[0];
    numeric_t scale = *((numeric_t *) d[1]);
    // numeric_t counts = *((numeric_t *) d[2]);
    numeric_t sigma = exp(x[0]);
    int nCodes = n - 1;

    /* Model distribution              p = exp(h) / sum(exp(h))  */
    numeric_t *p = (numeric_t *) malloc(nCodes * sizeof(numeric_t));
    for (int i = 0; i < nCodes; i++) p[i] = exp(x[i + 1]);
    numeric_t Z = 0;
    for (int i = 0; i < nCodes; i++) Z += p[i];
    numeric_t Zinv = 1.0 / Z;
    for (int i = 0; i < nCodes; i++) p[i] *= Zinv;

    /* Negative log likelihood     -loglk = -sum(C .* log(p)) */
    for (int i = 0; i < nCodes; i++) f -= C[i] * log(p[i]);
    free(p);

    /* Negative log prior */
    numeric_t ssq = 0;
    for (int i = 0; i < nCodes; i++) ssq += x[i + 1] * x[i + 1];
    f += (1.0 / (2.0 * sigma * sigma)) * ssq + nCodes * x[0];

    /* Negative log hyperprior */
    f += -x[0] + log(scale * scale + sigma * sigma);

    return f;
}

void HMCSiteHGrad(void *data, const numeric_t *x, numeric_t *g, const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a single-site hierarchical model */

    void **d = (void **)data;
    numeric_t *C = (numeric_t *) d[0];
    numeric_t scale = *((numeric_t *) d[1]);
    numeric_t counts = *((numeric_t *) d[2]);
    numeric_t sigma = exp(x[0]);
    int nCodes = n - 1;

    /* Model distribution              p = exp(h) / sum(exp(h))  */
    numeric_t *p = (numeric_t *) malloc(nCodes * sizeof(numeric_t));
    for (int i = 0; i < nCodes; i++) p[i] = exp(x[i + 1]);
    numeric_t Z = 0;
    for (int i = 0; i < nCodes; i++) Z += p[i];
    numeric_t Zinv = 1.0 / Z;
    for (int i = 0; i < nCodes; i++) p[i] *= Zinv;

    /* Negative log prior */
    numeric_t invSS = 1.0 / (sigma * sigma);
    numeric_t ssq = 0;
    for (int i = 0; i < nCodes; i++) ssq += x[i + 1] * x[i + 1];
    g[0] = -invSS * ssq + nCodes;
    for (int i = 0; i < nCodes; i++)
        g[i + 1] = -counts * (C[i] / counts - p[i]) + invSS * x[i + 1];

    /* Negative log hyperprior */
    g[0] += -1 + 2 * sigma * sigma / (scale * scale + sigma * sigma);

    free(p);
}

numeric_t HMCSiteHNonCenter(void *data, const numeric_t *x, const int n) {
    /* Computes the (unnormalized) negative log posterior of single-site 
       hierarchical model */
    numeric_t f = 0;

    void **d = (void **)data;
    numeric_t *C = (numeric_t *) d[0];
    numeric_t scale = *((numeric_t *) d[1]);
    // numeric_t counts = *((numeric_t *) d[2]);
    numeric_t sigma = exp(x[0]);
    int nCodes = n - 1;

    /* Model distribution              p = exp(h) / sum(exp(h))  */
    numeric_t *p = (numeric_t *) malloc(nCodes * sizeof(numeric_t));
    for (int i = 0; i < nCodes; i++) p[i] = exp(sigma * x[i + 1]);
    numeric_t Z = 0;
    for (int i = 0; i < nCodes; i++) Z += p[i];
    numeric_t Zinv = 1.0 / Z;
    for (int i = 0; i < nCodes; i++) p[i] *= Zinv;

    /* Negative log likelihood     -loglk = -sum(C .* log(p)) */
    for (int i = 0; i < nCodes; i++) f -= C[i] * log(p[i]);
    free(p);

    /* Negative log prior */
    numeric_t ssq = 0;
    for (int i = 0; i < nCodes; i++) ssq += x[i + 1] * x[i + 1];
    f += ssq / 2.0;

    /* Negative log hyperprior */
    f += -x[0] + log(scale * scale + sigma * sigma);

    return f;
}

void HMCSiteHGradNonCenter(void *data, const numeric_t *x, numeric_t *g, 
    const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a single-site hierarchical model */

    void **d = (void **)data;
    numeric_t *C = (numeric_t *) d[0];
    numeric_t scale = *((numeric_t *) d[1]);
    numeric_t counts = *((numeric_t *) d[2]);
    numeric_t sigma = exp(x[0]);
    int nCodes = n - 1;

    /* Model distribution              p = exp(h) / sum(exp(h))  */
    numeric_t *p = (numeric_t *) malloc(nCodes * sizeof(numeric_t));
    for (int i = 0; i < nCodes; i++) p[i] = exp(sigma * x[i + 1]);
    numeric_t Z = 0;
    for (int i = 0; i < nCodes; i++) Z += p[i];
    numeric_t Zinv = 1.0 / Z;
    for (int i = 0; i < nCodes; i++) p[i] *= Zinv;

    /* Negative log likelihood + log prior */
    numeric_t avH = 0;
    for (int i = 0; i < nCodes; i++) avH += p[i] * x[i + 1];
    numeric_t gradSum = 0;
    for (int i = 0; i < nCodes; i++) gradSum += C[i] * (x[i + 1] - avH);
    g[0] = -sigma * gradSum;

    for (int i = 0; i < nCodes; i++)
        g[i + 1] = sigma * counts * (p[i] - C[i] / counts) + x[i + 1];

    /* Negative log hyperprior */
    g[0] += -1 + 2 * sigma * sigma / (scale * scale + sigma * sigma);

    free(p);
}

void EstimatePairModelHMC(numeric_t *lambdas, numeric_t *x, alignment_t *ali, 
    options_t *options) {
    /* Infer the parameters and hyperparameters of an undirected graphical model
       with Gaussian priors over the couplings */

    void *data[2] = {(void *)ali, (void *)options};
    fprintf(stderr, "Initializing HMC\n");
    /* Tuned parameters for HMC */
    /* Number of variables */
    int n = ali->nParams + ali->nSites + ali->nSites * (ali->nSites - 1) / 2;
    int s = 100;                    /* Number of samples to collect */
    int L = 10;                     /* Integration number of steps */
    numeric_t eps = 0.1;            /* Median step size +/- 10-fold */
    int warmup = 1E2;               /* Number of samples to discard */

    /* Samples from HMC */
    numeric_t *X = (numeric_t *) malloc(n * s * sizeof(numeric_t));
    for (int ix = 0; ix < n * s; ix++) X[ix] = 0;
    numeric_t accRate =
        SampleHamiltonianMonteCarlo(HMCPairHNonCenter, HMCPairHGradNonCenter,
        data, X, n, s, L, &eps, warmup);

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Dump values to local directory */
    FILE *fpOutput = NULL;
    fpOutput = fopen("DUMP", "w");
    for (int ix = 0; ix < n * s; ix++) {
        double d = X[ix];
        fwrite(&d, sizeof(d), 1, fpOutput);
    }
    fprintf(stderr, "Parameter traces in DUMP: accRate = %f\n", accRate);

    /* Write single time trace */
    // for (int ix = 0; ix < s; ix++) {
    //     double d = X[n * ix];
    //     fwrite(&d, sizeof(d), 1, fpOutput);
    // }
    // fprintf(stderr, "Parameter trace in DUMP\n");

    fclose(fpOutput);
    exit(0);
    /* --------------------------------^DEBUG^--------------------------------*/
}

numeric_t HMCPairHNonCenter(void *data, const numeric_t *xB, const int n) {
    /* Computes the (unnormalized) negative log posterior of the pairwise
       hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    // options_t *options = (options_t *) d[1];

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate standard normal */
    // for (int i = 0; i < n; i++) f += (xB[i] - 5.0) * (xB[i] - 5.0) / 2.0;
    /* --------------------------------^DEBUG^--------------------------------*/

    /* Hyperparameters are indexed before parameter set */
    // const numeric_t *lambdas = xB;
    int offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    /* Parameter set is notated */
    const numeric_t *x = &(xB[offset]);

    /* Negative log-pseudolikelihood */
    numeric_t fx = 0;
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int a = 0; a < ali->nCodes; a++) siteH(i, a) = xHi(i, a);

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Compute potentials */
            for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));

            /* Conditional distribution given sequence background */
            numeric_t scale = H[0];
            for (int a = 1; a < ali->nCodes; a++)
                scale = (scale >= H[a] ? scale : H[a]);
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


            /* Log-likelihood contributions are scaled by sequence weight */
            numeric_t w = ali->weights[s];  
            siteFx -= w * log(P[seq(s, i)]);
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        fx += siteFx;
        free(Xi);
        }

        free(H);
        free(P);
    }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate standard normal */
    for (int i = 0; i < n; i++) fx += xB[i] * xB[i] / 2.0;
    /* ------------------------G-------^DEBUG^--------------------------------*/
    return fx;
}

void HMCPairHGradNonCenter(void *data, const numeric_t *xB, numeric_t *gB, 
    const int n) {
    /* Computes the gradient of the (unnormalized) negative log posterior of 
       a pairwise hierarchical model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    // options_t *options = (options_t *) d[1];

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate standard normal */
    // for (int i = 0; i < n; i++) gB[i] = xB[i] - 5.0;
    /* ------------------------G-------^DEBUG^--------------------------------*/

    /* Initialize gradient to zero */
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Hyperparameters are indexed before parameter set */
    // const numeric_t *lambdas = xB;
    // numeric_t *gLambdas = gB;
    int offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    /* Parameter set is notated */
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Negative log-pseudolikelihood */
    numeric_t fx = 0;
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    siteE(j, a, b) = xEij(i, j, a, b);
        for (int a = 0; a < ali->nCodes; a++) siteH(i, a) = xHi(i, a);

        numeric_t *Di = (numeric_t *) malloc(ali->nCodes * ali->nCodes
        * ali->nSites * sizeof(numeric_t));
        for (int d = 0; d < ali->nCodes * ali->nCodes * ali->nSites; d++)
            Di[d] = 0.0;

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Compute potentials */
            for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));

            /* Conditional distribution given sequence background */
            numeric_t scale = H[0];
            for (int a = 1; a < ali->nCodes; a++)
                scale = (scale >= H[a] ? scale : H[a]);
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


            /* Log-likelihood contributions are scaled by sequence weight */
            numeric_t w = ali->weights[s];  
            siteFx -= w * log(P[seq(s, i)]);

            /* Field gradient */
            siteDH(i, seq(s, i)) -= w;
            for (int a = 0; a < ali->nCodes; a++)
                siteDH(i, a) -= -w * P[a];

            /* Couplings gradient */
            int ix = seq(s, i);
            for (int j = 0; j < i; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = i + 1; j < ali->nSites; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        fx += siteFx;
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int a = 0; a < ali->nCodes; a++) dHi(i, a) += siteDH(i, a);
        free(Xi);
        free(Di);
        }

        free(H);
        free(P);
    }

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Test function: multivariate standard normal */
    for (int i = 0; i < n; i++) fx += xB[i] * xB[i] / 2.0;
    for (int i = 0; i < n; i++) gB[i] += xB[i];
    /* ------------------------G-------^DEBUG^--------------------------------*/


    /* Gaussian priors */
    // for (int i = 0; i < ali->nSites; i++)
    //     for (int ai = 0; ai < ali->nCodes; ai++) {
    //         dHi(i, ai) += lambdaHi(i) * 2.0 * xHi(i, ai);
    //         fx += lambdaHi(i) * xHi(i, ai) * xHi(i, ai);
    //     }

    // for (int i = 0; i < ali->nSites-1; i++)
    //     for (int j = i + 1; j < ali->nSites; j++)
    //         for (int ai = 0; ai < ali->nCodes; ai++)
    //             for (int aj = 0; aj < ali->nCodes; aj++) {
    //                 dEij(i, j, ai, aj) += lambdaEij(i, j)
    //                     * 2.0 * xEij(i, j, ai, aj);
    //                 fx += lambdaEij(i, j)
    //                     * xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
    //             }
    // return fx;
}

void EstimatePairModelMAP(numeric_t *x, numeric_t *lambdas, alignment_t *ali,
    options_t *options) {
    /* Computes Maximum a posteriori (MAP) estimates for the parameters of 
       an undirected graphical model by stochastic Maximum Likelihod */
    numeric_t eps = 0.01;
    numeric_t crit = 1E-3;

    /* Start timer */
    gettimeofday(&ali->start, NULL);

    /* Initialize Gibbs sampling with a random unconstrained sequences */
    init_genrand(42);
    ali->samples = (letter_t *)
        malloc(ali->nSites * options->gChains * sizeof(letter_t));
    for (int i = 0; i < ali->nSites * options->gChains; i++)
        ali->samples[i] = (genrand_int31() % ali->nCodes);

    /* --------------------------------_DEBUG_--------------------------------*/
    /* Warm start */
    /* Initialize L-BFGS */
    // lbfgs_parameter_t param;
    // lbfgs_parameter_init(&param);
    // param.epsilon = 1E-3;
    // param.max_iterations = 50;
    // void *d[3] = {(void *)ali, (void *)options, (void *)lambdas};
    // int ret = 0;
    // lbfgsfloatval_t fx;
    // ret = lbfgs(ali->nParams, x, &fx, PLMNegLogPosterior, ReportProgresslBFGS,
    //     (void*)d, &param);
    // fprintf(stderr, "Gradient optimization: %s\n", LBFGSErrorString(ret));
    // /* Presample from the pseudo-random number generator for thread safety */
    // int gSweeps = 100;
    // int gChains = options->gChains;
    // letter_t *sample = (letter_t *) malloc(gChains * gSweeps * ali->nSites
    //     * sizeof(letter_t));
    // int nSteps = gChains * gSweeps * ali->nSites;
    // int *siteI = (int *) malloc(nSteps * sizeof(int));
    // double *codeU = (double *) malloc(nSteps * sizeof(double));
    // for (int i = 0; i < nSteps; i++) siteI[i] = genrand_int31() % ali->nSites;
    // for (int i = 0; i < nSteps; i++) codeU[i] = genrand_real3();
    // #pragma omp parallel for
    // for (int c = 0; c < gChains; c++) {
    //     /* Samples gSweeps sequences in each chain */
    //     numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
    //     for (int s = 0; s < gSweeps; s++) {
    //         /* Sweep nSites positions */
    //         for (int sx = 0; sx < ali->nSites; sx++) {
    //             /* Pick a random site */
    //             int i = siteI[c * gSweeps * ali->nSites + s * ali->nSites + sx];

    //             /* Compute conditional CDF at the site */
    //             for (int a = 0; a < ali->nCodes; a++)
    //                 P[a] = exp(lambdaHi(i)) * xHi(i, a);
    //             for (int j = 0; j < i; j++)
    //                 for (int a = 0; a < ali->nCodes; a++)
    //                     P[a] += exp(lambdaEij(i, j))
    //                          * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
    //             for (int j = i + 1; j < ali->nSites; j++)
    //                 for (int a = 0; a < ali->nCodes; a++)
    //                     P[a] += exp(lambdaEij(i, j))
    //                          * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
    //             numeric_t scale = P[0];
    //             for (int a = 1; a < ali->nCodes; a++)
    //                 scale = (scale >= P[a] ? scale : P[a]);
    //             for (int a = 0; a < ali->nCodes; a++) P[a] = exp(P[a] - scale);
    //             for (int a = 1; a < ali->nCodes; a++) P[a] += P[a - 1];

    //             /* Choose a new code for the site */
    //             double u = P[ali->nCodes - 1] *
    //                 codeU[c * gSweeps * ali->nSites + s * ali->nSites + sx];
    //             int aNew = 0;
    //             while (u > P[aNew]) aNew++;
    //             ali->samples[c * ali->nSites + i] = aNew;
    //         }
    //     }
    //     free(P);
    // }
    // free(siteI);
    // free(codeU);
    /* --------------------------------^DEBUG^--------------------------------*/


    /* Array of void pointers provides relevant data structures */
    void *data[3] = {(void *)ali, (void *)options, (void *)lambdas};

    EstimateMaximumAPosteriori(MAPPairGibbs, data, x, ali->nParams, eps,
        options->maxIter, crit);
}

void MAPPairGibbs(void *data, const numeric_t *x, numeric_t *g, const int n) {
    /* Computes the gradient of the negative log posterior of a Potts model */
    void **d = (void **)data;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];
    const numeric_t *lambdas = d[2];

    /* Initialize gradient to zero */
    for (int i = 0; i < n; i++) g[i] = 0;

    /* Gradient: sitewise marginals of the data */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            dHi(i, ai) = -ali->nEff * fi(i, ai);

    /* Gradient: pairwise marginals of the data */
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    dEij(i, j, ai, aj) = -ali->nEff * fij(i, j, ai, aj);

    /* Gradient: marginals of the model by persistent Markov chains */
    int gSweeps = options->gSweeps;
    int gChains = options->gChains;
    letter_t *sample = (letter_t *) malloc(gChains * gSweeps * ali->nSites
        * sizeof(letter_t));

    /* Presample from the pseudo-random number generator for thread safety */
    int nSteps = gChains * gSweeps * ali->nSites;
    int *siteI = (int *) malloc(nSteps * sizeof(int));
    double *codeU = (double *) malloc(nSteps * sizeof(double));
    for (int i = 0; i < nSteps; i++) siteI[i] = genrand_int31() % ali->nSites;
    for (int i = 0; i < nSteps; i++) codeU[i] = genrand_real3();

    /* Parallelize across the chains */
    #pragma omp parallel for
    for (int c = 0; c < gChains; c++) {
        /* Samples gSweeps sequences in each chain */
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        for (int s = 0; s < gSweeps; s++) {
            /* Sweep nSites positions */
            for (int sx = 0; sx < ali->nSites; sx++) {
                /* Pick a random site */
                int i = siteI[c * gSweeps * ali->nSites + s * ali->nSites + sx];

                /* Compute conditional CDF at the site */
                for (int a = 0; a < ali->nCodes; a++)
                    P[a] = exp(lambdaHi(i)) * xHi(i, a);
                for (int j = 0; j < i; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        P[a] += exp(lambdaEij(i, j))
                             * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
                for (int j = i + 1; j < ali->nSites; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        P[a] += exp(lambdaEij(i, j))
                             * xEij(i, j, a, ali->samples[c * ali->nSites + j]);
                numeric_t scale = P[0];
                for (int a = 1; a < ali->nCodes; a++)
                    scale = (scale >= P[a] ? scale : P[a]);
                for (int a = 0; a < ali->nCodes; a++) P[a] = exp(P[a] - scale);
                for (int a = 1; a < ali->nCodes; a++) P[a] += P[a - 1];

                /* Choose a new code for the site */
                double u = P[ali->nCodes - 1] *
                    codeU[c * gSweeps * ali->nSites + s * ali->nSites + sx];
                int aNew = 0;
                while (u > P[aNew]) aNew++;
                ali->samples[c * ali->nSites + i] = aNew;
            }

            /* Copy sequence into the global sample */
            for (int i = 0; i < ali->nSites; i++)
                sample[c * gSweeps * ali->nSites + s * ali->nSites + i] =
                    ali->samples[c * ali->nSites + i];
        }
        free(P);
    }

    /* Contribute to global gradient for centered parameters */
    numeric_t nRatio =
        ((numeric_t) ali->nEff) / ((numeric_t) (gChains * gSweeps));
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++)
        for (int c = 0; c < gChains; c++)
            for (int s = 0; s < gSweeps; s++)
                dHi(i, sample[c * gSweeps * ali->nSites + s * ali->nSites + i])
                     += nRatio;
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int c = 0; c < gChains; c++)
                for (int s = 0; s < gSweeps; s++)
                    dEij(i, j, 
                        sample[c * gSweeps * ali->nSites + s * ali->nSites + i],
                        sample[c * gSweeps * ali->nSites + s * ali->nSites + j])
                        += nRatio;
    free(sample);
    free(siteI);
    free(codeU);

    numeric_t fx = 0;
    fx = AddPriorsCentered(x, g, lambdas, fx, ali, options);
}

void EstimatePairModelPLM(numeric_t *x, numeric_t *lambdas, alignment_t *ali,
    options_t *options) {
    /* Computes Maximum a posteriori (MAP) estimates for the parameters of 
       and undirected graphical model by L-BFGS */

    /* Start timer */
    gettimeofday(&ali->start, NULL);

    /* Initialize L-BFGS */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.epsilon = 1E-3;
    param.max_iterations = options->maxIter; /* 0 is unbounded */

    /* Array of void pointers provides relevant data structures */
    void *d[3] = {(void *)ali, (void *)options, (void *)lambdas};

    /* Estimate parameters by optimization */
    static lbfgs_evaluate_t algo;
    switch(options->estimatorMAP) {
        case INFER_MAP_PLM:
            algo = PLMNegLogPosterior;
            break;
        case INFER_MAP_PLM_GAPREDUCE:
            algo = PLMNegLogPosteriorGapReduce;
            break;
        case INFER_MAP_PLM_BLOCK:
            algo = PLMNegLogPosteriorBlock;
            break;
        case INFER_MAP_PLM_DROPOUT:
            algo = PLMNegLogPosteriorDO;
            break;
        default:
            algo = PLMNegLogPosterior;
    }

    if (options->zeroAPC == 1) fprintf(stderr,
            "Estimating coupling hyperparameters le = 1/2 inverse variance\n");

    int ret = 0;
    lbfgsfloatval_t fx;
    ret = lbfgs(ali->nParams, x, &fx, algo, ReportProgresslBFGS,
        (void*)d, &param);
    fprintf(stderr, "Gradient optimization: %s\n", LBFGSErrorString(ret));

    /* Optionally re-estimate parameters with adjusted hyperparameters */
    if (options->zeroAPC == 1) {
        /* Form new priors on the variances */
        ZeroAPCPriors(ali, options, lambdas, x);

        /* Reinitialize coupling parameters */
        for (int i = 0; i < ali->nSites - 1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        xEij(i, j, ai, aj) = 0.0;

        /* Iterate estimation with new hyperparameter estimates */
        options->zeroAPC = 2;
        ret = lbfgs(ali->nParams, x, &fx, algo,
            ReportProgresslBFGS, (void*)d, &param);
        fprintf(stderr, "Gradient optimization: %s\n", LBFGSErrorString(ret));
    }
}

static lbfgsfloatval_t PLMNegLogPosterior(void *instance,
    const lbfgsfloatval_t *xB, lbfgsfloatval_t *gB, const int n,
    const lbfgsfloatval_t step) {
    /* Compute the the negative log posterior, which is the negative 
       penalized log-(pseudo)likelihood and the objective for MAP inference
    */
    void **d = (void **)instance;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];
    numeric_t *lambdas = (numeric_t *) d[2];

    /* Initialize log-likelihood and gradient */
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Optionally optimize hyperparameters for noncentered parameterizations */
    numeric_t *gLambdas = NULL;
    int offset = 0;
    if (options->noncentered) {
        /* Hyperparameters are now variables */
        lambdas = (numeric_t *) xB;
        gLambdas = gB;
        offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    }
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Profiling code START */
    #if defined(PROFILE_TIMES)
        struct timeval tic;
        gettimeofday(&tic, NULL);
    #endif

    /* Negative log-pseudolikelihood */
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        if (options->noncentered) {
            /* Noncentered parameterization */
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = exp(lambdaEij(i, j))
                                         * xEij(i, j, a, b);
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = exp(lambdaEij(i, j))
                                         * xEij(i, j, a, b);
            for (int a = 0; a < ali->nCodes; a++)
                siteH(i, a) = exp(lambdaHi(i)) * xHi(i, a);
        } else {
            /* Centered parameterization */
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = xEij(i, j, a, b);
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = xEij(i, j, a, b);
            for (int a = 0; a < ali->nCodes; a++) siteH(i, a) = xHi(i, a);
        }
        
        numeric_t *Di = (numeric_t *) malloc(ali->nCodes * ali->nCodes
        * ali->nSites * sizeof(numeric_t));
        for (int d = 0; d < ali->nCodes * ali->nCodes * ali->nSites; d++)
            Di[d] = 0.0;

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Compute potentials */
            for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    H[a] += siteE(j, a, seq(s, j));

            /* Conditional distribution given sequence background */
            numeric_t scale = H[0];
            for (int a = 1; a < ali->nCodes; a++)
                scale = (scale >= H[a] ? scale : H[a]);
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


            /* Log-likelihood contributions are scaled by sequence weight */
            numeric_t w = ali->weights[s];	
            siteFx -= w * log(P[seq(s, i)]);

            /* Field gradient */
            siteDH(i, seq(s, i)) -= w;
            for (int a = 0; a < ali->nCodes; a++)
                siteDH(i, a) -= -w * P[a];

            /* Couplings gradient */
            int ix = seq(s, i);
            for (int j = 0; j < i; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = i + 1; j < ali->nSites; j++)
                siteDE(j, ix, seq(s, j)) -= w;
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    siteDE(j, a, seq(s, j)) -= -w * P[a];
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        fx += siteFx;
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int a = 0; a < ali->nCodes; a++) dHi(i, a) += siteDH(i, a);
        free(Xi);
        free(Di);
        }

        free(H);
        free(P);
    }

    /* Transform gradients for noncentered parameterization */
    if (options->noncentered) {
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                gLambdaHi(i) += exp(lambdaHi(i)) * xHi(i, ai) * dHi(i, ai);
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        gLambdaEij(i,j) += exp(lambdaEij(i, j))
                                            * xEij(i, j, ai, aj)
                                            * dEij(i, j, ai, aj);
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                dHi(i, ai) *= exp(lambdaHi(i));
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        dEij(i, j, ai, aj) *= exp(lambdaEij(i, j));
    }

    /* Profiling code STOP */
    #if defined(PROFILE_TIMES)
        struct timeval toc;
        gettimeofday(&toc, NULL);
        /* Subtraction routines from FastTree */
        if (toc.tv_usec < tic.tv_usec) {
            int nsec = (tic.tv_usec - toc.tv_usec) / 1000000 + 1;
            tic.tv_usec -= 1000000 * nsec;
            tic.tv_sec += nsec;
        }
        if (toc.tv_usec - tic.tv_usec > 1000000) {
            int nsec = (toc.tv_usec - tic.tv_usec) / 1000000;
            tic.tv_usec += 1000000 * nsec;
            tic.tv_sec -= nsec;
        }
        int usec = (int) (toc.tv_usec - tic.tv_usec);
        int sec = (int) (toc.tv_sec - tic.tv_sec);
        fprintf(stderr, " %i.%2i seconds\n", sec, usec/ 10000);
    #endif

    ali->negLogLk = fx / ali->nEff;

    if (options->noncentered) {
        fx = AddPriorsNoncentered(x, g, lambdas, gLambdas, fx, ali, options);
    } else {
        fx = AddPriorsCentered(x, g, lambdas, fx, ali, options);
    }

    /* Scale the function and gradient */
    // numeric_t invNEff = 1.0 / ali->nEff;
    // fx *= invNEff;
    // for (int i = 0; i < n; i++) gB[i] *= invNEff;

    return fx;
}

static lbfgsfloatval_t PLMNegLogPosteriorGapReduce(void *instance,
    const lbfgsfloatval_t *xB, lbfgsfloatval_t *gB, const int n,
    const lbfgsfloatval_t step) {
    /* Compute the the negative log posterior, which is the negative 
       penalized log-(pseudo)likelihood and the objective for MAP inference
    */
    void **d = (void **)instance;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];
    numeric_t *lambdas = (numeric_t *) d[2];

    /* Initialize log-likelihood and gradient */
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < n; i++) gB[i] = 0;

    /* Optionally optimize hyperparameters for noncentered parameterizations */
    numeric_t *gLambdas = NULL;
    int offset = 0;
    if (options->noncentered == 1) {
        /* Hyperparameters are now variables */
        lambdas = (numeric_t *) xB;
        gLambdas = gB;
        offset = ali->nSites + ali->nSites * (ali->nSites - 1) / 2; 
    }
    const numeric_t *x = &(xB[offset]);
    numeric_t *g = &(gB[offset]);

    /* Profiling code START */
    #if defined(PROFILE_TIMES)
        struct timeval tic;
        gettimeofday(&tic, NULL);
    #endif

    /* Negative log-pseudolikelihood */
    #pragma omp parallel for
    for (int i = 0; i < ali->nSites; i++) {
        numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
        numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));

        numeric_t siteFx = 0.0;
        /* Reshape site parameters and gradient into local blocks */
        numeric_t *Xi = (numeric_t *) malloc(ali->nCodes * ali->nCodes
            * ali->nSites * sizeof(numeric_t));
        if (options->noncentered) {
            /* Noncentered parameterization */
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = exp(lambdaEij(i, j))
                                         * xEij(i, j, a, b);
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = exp(lambdaEij(i, j))
                                         * xEij(i, j, a, b);
            for (int a = 0; a < ali->nCodes; a++)
                siteH(i, a) = exp(lambdaHi(i)) * xHi(i, a);
        } else {
            /* Centered parameterization */
            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = xEij(i, j, a, b);
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    for (int b = 0; b < ali->nCodes; b++)
                        siteE(j, a, b) = xEij(i, j, a, b);
            for (int a = 0; a < ali->nCodes; a++) siteH(i, a) = xHi(i, a);
        }

        numeric_t *Di = (numeric_t *) malloc(ali->nCodes * ali->nCodes
        * ali->nSites * sizeof(numeric_t));
        for (int d = 0; d < ali->nCodes * ali->nCodes * ali->nSites; d++)
            Di[d] = 0.0;

        /* Site negative conditional log likelihoods */
        for (int s = 0; s < ali->nSeqs; s++) {
            /* Only ungapped sites are considered in the model */
            if (seq(s, i) >= 0) {
                /* Compute potentials */
                for (int a = 0; a < ali->nCodes; a++) H[a] = siteH(i, a);
                for (int j = 0; j < i; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        if (seq(s, j) >= 0)
                            H[a] += siteE(j, a, seq(s, j));
                for (int j = i + 1; j < ali->nSites; j++)
                    for (int a = 0; a < ali->nCodes; a++)
                        if (seq(s, j) >= 0)
                            H[a] += siteE(j, a, seq(s, j));

                /* Conditional distribution given sequence background */
                numeric_t scale = H[0];
                for (int a = 1; a < ali->nCodes; a++)
                    scale = (scale >= H[a] ? scale : H[a]);
                for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a] - scale);
                numeric_t Z = 0;
                for (int a = 0; a < ali->nCodes; a++) Z += P[a];
                numeric_t Zinv = 1.0 / Z;
                for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;


                /* Log-likelihood contributions are scaled by sequence weight */
                numeric_t w = ali->weights[s];  
                siteFx -= w * log(P[seq(s, i)]);

                /* Field gradient */
                siteDH(i, seq(s, i)) -= w;
                for (int a = 0; a < ali->nCodes; a++)
                    siteDH(i, a) -= -w * P[a];

                /* Couplings gradient */
                int ix = seq(s, i);
                for (int j = 0; j < i; j++)
                    if (seq(s, j) >= 0)
                        siteDE(j, ix, seq(s, j)) -= w;
                for (int j = i + 1; j < ali->nSites; j++)
                    if (seq(s, j) >= 0)
                        siteDE(j, ix, seq(s, j)) -= w;
                for (int j = 0; j < i; j++)
                    if (seq(s, j) >= 0)
                        for (int a = 0; a < ali->nCodes; a++)
                            siteDE(j, a, seq(s, j)) -= -w * P[a];
                for (int j = i + 1; j < ali->nSites; j++)
                    if (seq(s, j) >= 0)
                        for (int a = 0; a < ali->nCodes; a++)
                            siteDE(j, a, seq(s, j)) -= -w * P[a];
            }
        }

        /* Contribute local loglk and gradient to global */
        #pragma omp critical
        {
        fx += siteFx;
        for (int j = 0; j < i; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int j = i + 1; j < ali->nSites; j++)
            for (int a = 0; a < ali->nCodes; a++)
                for (int b = 0; b < ali->nCodes; b++)
                    dEij(i, j, a, b) += siteDE(j, a, b);
        for (int a = 0; a < ali->nCodes; a++) dHi(i, a) += siteDH(i, a);
        free(Xi);
        free(Di);
        }

        free(H);
        free(P);
    }

    /* Transform gradients for noncentered parameterization */
    if (options->noncentered) {
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                gLambdaHi(i) += exp(lambdaHi(i)) * xHi(i, ai) * dHi(i, ai);
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        gLambdaEij(i,j) += exp(lambdaEij(i, j))
                                            * xEij(i, j, ai, aj)
                                            * dEij(i, j, ai, aj);
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                dHi(i, ai) *= exp(lambdaHi(i));
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++)
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        dEij(i, j, ai, aj) *= exp(lambdaEij(i, j));
    }

    /* Profiling code STOP */
    #if defined(PROFILE_TIMES)
        struct timeval toc;
        gettimeofday(&toc, NULL);
        /* Subtraction routines from FastTree */
        if (toc.tv_usec < tic.tv_usec) {
            int nsec = (tic.tv_usec - toc.tv_usec) / 1000000 + 1;
            tic.tv_usec -= 1000000 * nsec;
            tic.tv_sec += nsec;
        }
        if (toc.tv_usec - tic.tv_usec > 1000000) {
            int nsec = (toc.tv_usec - tic.tv_usec) / 1000000;
            tic.tv_usec += 1000000 * nsec;
            tic.tv_sec -= nsec;
        }
        int usec = (int) (toc.tv_usec - tic.tv_usec);
        int sec = (int) (toc.tv_sec - tic.tv_sec);
        fprintf(stderr, " %i.%2i seconds\n", sec, usec/ 10000);
    #endif

    ali->negLogLk = fx;

    if (options->noncentered) {
        fx = AddPriorsNoncentered(x, g, lambdas, gLambdas, fx, ali, options);
    } else {
        fx = AddPriorsCentered(x, g, lambdas, fx, ali, options);
    }
    return fx;
}

static lbfgsfloatval_t PLMNegLogPosteriorBlock(void *instance,
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n,
    const lbfgsfloatval_t step) {
    /* Compute the the negative log posterior, which is the negative 
       penalized log-(pseudo)likelihood and the objective for MAP inference
    */
    void **d = (void **)instance;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];
    numeric_t *lambdas = (numeric_t *) d[2];

    /* Initialize log-likelihood and gradient */
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < ali->nParams; i++) g[i] = 0;

    /* Profiling code START */
    #if defined(PROFILE_TIMES)
        struct timeval tic;
        gettimeofday(&tic, NULL);
    #endif

    /* Block fields hi */
    numeric_t *hi = (numeric_t *)
        malloc(ali->nSites * ali->nCodes * sizeof(numeric_t));
    numeric_t *gHi = (numeric_t *)
        malloc(ali->nSites * ali->nCodes * sizeof(numeric_t));
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) Hi(i, ai) = xHi(i, ai);
    for (int i = 0; i < ali->nSites * ali->nCodes; i++) gHi[i] = 0;

    /* Block couplings eij */
    numeric_t *eij = (numeric_t *) malloc(ali->nSites * ali->nSites
        * ali->nCodes * ali->nCodes * sizeof(numeric_t));
    numeric_t *gEij = (numeric_t *) malloc(ali->nSites * ali->nSites
        * ali->nCodes * ali->nCodes * sizeof(numeric_t));
    for (int i = 0; i < ali->nSites * ali->nSites * ali->nCodes * ali->nCodes;
        i++) eij[i] = 0.0;
    for (int i = 0; i < ali->nSites * ali->nSites * ali->nCodes * ali->nCodes;
        i++) gEij[i] = 0.0;
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    Eij(j, aj, i, ai) = Eij(i, ai, j, aj) = xEij(i, j, ai, aj);


    /* Negative log-pseudolikelihood */
    for (int s = 0; s < ali->nSeqs; s++) {
        /* Form potential for conditional log likelihoods at every site */
        numeric_t *H = (numeric_t *)
            malloc(ali->nCodes * ali->nSites * sizeof(numeric_t));
        numeric_t *Z = (numeric_t *) malloc(ali->nSites * sizeof(numeric_t));

        /* Initialize potentials with fields */
        // memcpy(H, hi, ali->nSites * ali->nCodes * sizeof(numeric_t));
        for(int jx = 0; jx < ali->nSites * ali->nCodes; jx++) H[jx] = hi[jx];

        /* Contribute coupling block due to i, ai */
        for (int i = 0; i < ali->nSites; i++) {
            const letter_t ai = seq(s, i);
            const numeric_t *jB = &(Eij(i, ai, 0, 0));
            for(int jx = 0; jx < ali->nSites * ali->nCodes; jx++)
                H[jx] += jB[jx];
        }

        /* Conditional log likelihoods */
        for (int i = 0; i < ali->nSites * ali->nCodes; i++) H[i] = exp(H[i]);
        for (int i = 0; i < ali->nSites; i++) Z[i] = 0;
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nSites; ai++) Z[i] += Hp(i, ai);
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nSites; ai++) Hp(i, ai) /= Z[i];

        numeric_t seqFx = 0;
        for (int i = 0; i < ali->nSites; i++)
            seqFx -= ali->weights[s] * log(Hp(i, seq(s, i)));

        for(int jx = 0; jx < ali->nSites * ali->nCodes; jx++)
            H[jx] *= -ali->weights[s];

        for (int i = 0; i < ali->nSites; i++)
            gHi(i, seq(s, i)) -= ali->weights[s];
        for(int jx = 0; jx < ali->nSites * ali->nCodes; jx++) gHi[jx] -= H[jx];

        for (int i = 0; i < ali->nSites - 1; i++)
            for (int j = i; j < ali->nSites; j++)
                gEij(i, seq(s, i), j, seq(s, j)) -= ali->weights[s];

        for (int i = 0; i < ali->nSites; i++) {
            const letter_t ai = seq(s, i);
            numeric_t *jgBlock = &(gEij(i, ai, 0, 0));
            for (int jx = 0; jx < ali->nSites * ali->nCodes; jx++)
                jgBlock[jx] -= H[jx];
        }

        free(H);
        free(Z);
        fx += seqFx;
    }

    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++)
            dHi(i, ai) += gHi(i, ai);

    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    dEij(i, j, ai, aj) += gEij(j, aj, i, ai) + gEij(i, ai, j, aj);
    free(hi);
    free(gHi);
    free(eij);
    free(gEij);

    /* Profiling code STOP */
    #if defined(PROFILE_TIMES)
        struct timeval toc;
        gettimeofday(&toc, NULL);
        /* Subtraction routines from FastTree */
        if (toc.tv_usec < tic.tv_usec) {
            int nsec = (tic.tv_usec - toc.tv_usec) / 1000000 + 1;
            tic.tv_usec -= 1000000 * nsec;
            tic.tv_sec += nsec;
        }
        if (toc.tv_usec - tic.tv_usec > 1000000) {
            int nsec = (toc.tv_usec - tic.tv_usec) / 1000000;
            tic.tv_usec += 1000000 * nsec;
            tic.tv_sec -= nsec;
        }
        int usec = (int) (toc.tv_usec - tic.tv_usec);
        int sec = (int) (toc.tv_sec - tic.tv_sec);
        fprintf(stderr, " %i.%2i seconds\n", sec, usec/ 10000);
    #endif

    ali->negLogLk = fx;

    fx = AddPriorsCentered(x, g, lambdas, fx, ali, options);
    return fx;
}

static lbfgsfloatval_t PLMNegLogPosteriorDO(void *instance,
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n,
    const lbfgsfloatval_t step) {
    /* Compute the the negative log posterior, which is the negative 
       penalized log-(pseudo)likelihood and the objective for MAP inference
    */
    void **d = (void **)instance;
    alignment_t *ali = (alignment_t *) d[0];
    options_t *options = (options_t *) d[1];
    numeric_t *lambdas = (numeric_t *) d[2];

    /* Initialize log-likelihood and gradient */
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < ali->nParams; i++) g[i] = 0;

    /* Profiling code START */
    #if defined(PROFILE_TIMES)
        struct timeval tic;
        gettimeofday(&tic, NULL);
    #endif

    numeric_t *H = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
    numeric_t *P = (numeric_t *) malloc(ali->nCodes * sizeof(numeric_t));
    int *drop_mask = (int *) malloc(ali->nParams * sizeof(int));
    for (int s = 0; s < ali->nSeqs; s++) {
        /* Generate random bit mask over parameters */
        for (int p = 0; p < ali->nParams; p ++)
            drop_mask[p] = (int) rand() % 2;

        /* Pseudolikelihood objective */
        for (int i = 0; i < ali->nSites; i++) {
            for (int a = 0; a < ali->nCodes; a++) H[a] = bitHi(i, a)
                                               * xHi(i, a);
            for (int a = 0; a < ali->nCodes; a++)
                for (int j = 0; j < i; j++)
                    H[a] += bitEij(i, j, a, seq(s, j))
                            * xEij(i, j, a, seq(s, j));
            for (int a = 0; a < ali->nCodes; a++)
                for (int j = i + 1; j < ali->nSites; j++)
                    H[a] += bitEij(i, j, a, seq(s, j))
                            * xEij(i, j, a, seq(s, j));

            /* Compute distribution from potential */
            for (int a = 0; a < ali->nCodes; a++) P[a] = exp(H[a]);
            numeric_t Z = 0;
            for (int a = 0; a < ali->nCodes; a++) Z += P[a];
            numeric_t Zinv = 1.0 / Z;
            for (int a = 0; a < ali->nCodes; a++) P[a] *= Zinv;

            /* Log-likelihood contributions */
            fx -= ali->weights[s] * log(P[seq(s, i)]);

            /* Field gradient */
            dHi(i, seq(s, i)) -= bitHi(i, seq(s, i)) * ali->weights[s];
            for (int a = 0; a < ali->nCodes; a++)
                dHi(i, a) -= -bitHi(i, a) * ali->weights[s] * P[a];

            /* Couplings gradient */
            for (int j = 0; j < i; j++)
                dEij(i, j, seq(s, i), seq(s, j)) -=
                    bitEij(i, j, seq(s, i), seq(s, j)) * ali->weights[s];
            for (int j = i + 1; j < ali->nSites; j++)
                dEij(i, j, seq(s, i), seq(s, j)) -=
                    bitEij(i, j, seq(s, i), seq(s, j)) * ali->weights[s];

            for (int j = 0; j < i; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    dEij(i, j, a, seq(s, j)) -=
                        -bitEij(i, j, a, seq(s, j)) * ali->weights[s] * P[a];
            for (int j = i + 1; j < ali->nSites; j++)
                for (int a = 0; a < ali->nCodes; a++)
                    dEij(i, j, a, seq(s, j)) -=
                        -bitEij(i, j, a, seq(s, j)) * ali->weights[s] * P[a];
        }
    }
    free(H);
    free(P);
    free(drop_mask);

    /* Profiling code STOP */
    #if defined(PROFILE_TIMES)
        struct timeval toc;
        gettimeofday(&toc, NULL);
        /* Subtraction routines from FastTree */
        if (toc.tv_usec < tic.tv_usec) {
            int nsec = (tic.tv_usec - toc.tv_usec) / 1000000 + 1;
            tic.tv_usec -= 1000000 * nsec;
            tic.tv_sec += nsec;
        }
        if (toc.tv_usec - tic.tv_usec > 1000000) {
            int nsec = (toc.tv_usec - tic.tv_usec) / 1000000;
            tic.tv_usec += 1000000 * nsec;
            tic.tv_sec -= nsec;
        }
        int usec = (int) (toc.tv_usec - tic.tv_usec);
        int sec = (int) (toc.tv_sec - tic.tv_sec);
        fprintf(stderr, " %i.%2.3i seconds\n", sec, usec/ 10000);
    #endif

    ali->negLogLk = fx;

    fx = AddPriorsCentered(x, g, lambdas, fx, ali, options);
    return fx;
}

static int ReportProgresslBFGS(void *instance, const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step, int n, int k, int ls) {
    void **d = (void **)instance;
    alignment_t *ali = (alignment_t *)d[0];

    /* Compute norms of relevant parameters */
    lbfgsfloatval_t hNorm = 0.0, eNorm = 0.0, hGNorm = 0.0, eGNorm = 0.0;
    for (int i = 0; i < ali->nSites * ali->nCodes; i++)
        hNorm += x[i]*x[i];
    for (int i = 0; i < ali->nSites * ali->nCodes; i++)
        hGNorm += g[i]*g[i];
    for (int i = ali->nSites * ali->nCodes; i < ali->nParams; i++)
        eNorm += x[i]*x[i];
    for (int i = ali->nSites * ali->nCodes; i < ali->nParams; i++)
        eGNorm += g[i]*g[i];
    hNorm = sqrt(hNorm);
    hGNorm = sqrt(hGNorm);
    eNorm = sqrt(eNorm);
    eGNorm = sqrt(eGNorm);

    /* Retrieve elapsed time */
    static struct timeval now;
    gettimeofday(&now, NULL);
    if (now.tv_usec < ali->start.tv_usec) {
        int nsec = (ali->start.tv_usec - now.tv_usec) / 1000000 + 1;
        ali->start.tv_usec -= 1000000 * nsec;
        ali->start.tv_sec += nsec;
    }
    if (now.tv_usec - ali->start.tv_usec > 1000000) {
        int nsec = (now.tv_usec - ali->start.tv_usec) / 1000000;
        ali->start.tv_usec += 1000000 * nsec;
        ali->start.tv_sec -= nsec;
    }
    numeric_t elapsed = (numeric_t) (now.tv_sec - ali->start.tv_sec)
                      + ((numeric_t) (now.tv_usec - ali->start.tv_usec)) / 1E6;

    if (k == 1) fprintf(stderr,
        "iter\ttime\tcond\tfx\t-loglk"
        "\t||h||\t||e||\n");
    fprintf(stderr, "%d\t%.1f\t%.2f\t%.1f\t%.1f\t%.1f\t%.1f\n",
        k, elapsed, gnorm / xnorm, fx, ali->negLogLk, hNorm, eNorm);
    return 0;
}

void PreCondition(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, alignment_t *ali, options_t *options) {
    /* Currently empty */
}

numeric_t AddPriorsCentered(const numeric_t *x, numeric_t *g, 
    const numeric_t *lambdas, numeric_t fx, alignment_t *ali, options_t *options) {
    if (options->zeroAPC == 1)
        for (int i = 0; i < ali->nSites; i++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                dHi(i, ai) = 0.0;

    /* Gaussian priors */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) {
            dHi(i, ai) += lambdaHi(i) * 2.0 * xHi(i, ai);
            fx += lambdaHi(i) * xHi(i, ai) * xHi(i, ai);
        }
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++) {
                    dEij(i, j, ai, aj) += lambdaEij(i, j)
                        * 2.0 * xEij(i, j, ai, aj);
                    fx += lambdaEij(i, j)
                        * xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
                }

    /* Group (L1/L2) regularization  */
    if (options->lambdaGroup > 0)
        for (int i = 0; i < ali->nSites - 1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                double l2 = REGULARIZATION_GROUP_EPS;
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        l2 += xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
                double l1 = sqrt(l2);
                fx += options->lambdaGroup * l1;
                for (int ai = 0; ai < ali->nCodes; ai++)
                    for (int aj = 0; aj < ali->nCodes; aj++)
                        dEij(i, j, ai, aj) += options->lambdaGroup * xEij(i, j, ai, aj) / l1;
            }

    return fx;
}

numeric_t AddPriorsNoncentered(const numeric_t *x, numeric_t *g, 
    const numeric_t *lambdas, numeric_t *gLambdas, numeric_t fx,
    alignment_t *ali, options_t *options) {

    /* Standard normal priors */
    for (int i = 0; i < ali->nSites; i++)
        for (int ai = 0; ai < ali->nCodes; ai++) {
            dHi(i, ai) += xHi(i, ai);
            fx += 0.5 * xHi(i, ai) * xHi(i, ai);
        }
    for (int i = 0; i < ali->nSites-1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++) {
                    dEij(i, j, ai, aj) += xEij(i, j, ai, aj);
                    fx += 0.5 * xEij(i, j, ai, aj) * xEij(i, j, ai, aj);
                }
    /* Prior for the variances */
    numeric_t scaleH = options->scaleH;
    numeric_t scaleE = options->scaleE;
    if (options->hyperprior == PRIOR_HALFCAUCHYPLUS) {
        /* The Horseshoe+: Double Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            numeric_t sigmaHi = exp(lambdaHi(i));
            if (abs(scaleH - sigmaHi) < 1E-3) {
                /* Numerically stable computation as sigma -> scale */
                fx += -log(2.0 / (PI * PI));
            } else {
                /* Double indicator functions negate negative logs */
                numeric_t I = (numeric_t) (2 * (scaleH > sigmaHi) - 1);
                fx += -log(4.0 / (PI * PI)) - log(scaleH) - lambdaHi(i)
                           - log(I * (log(scaleH) - lambdaHi(i)))
                           + log(I * (scaleH * scaleH - exp(2 * lambdaHi(i))));
                gLambdaHi(i) += - 1.0
                                + 1.0 / (log(scaleH) - lambdaHi(i))
                                - 2.0 * exp(2 * lambdaHi(i))
                                  / (scaleH * scaleH - exp(2 * lambdaHi(i)));
            }
        }
        for (int i = 0; i < ali->nSites - 1; i++) 
            for (int j = i + 1; j < ali->nSites; j++) {
                numeric_t sigmaEij = exp(lambdaEij(i, j));
                if (abs(scaleE - sigmaEij) < 1E-3) {
                    /* Numerically stable computation as sigma -> scale */
                    fx += -log(2.0 / (PI * PI));
                } else {
                    /* Double indicator functions negate negative logs */
                    numeric_t I = (numeric_t) (2 * (scaleE > sigmaEij) - 1);
                    fx += -log(4.0 / (PI * PI)) - log(scaleE) - lambdaEij(i, j)
                               - log(I * (log(scaleE) - lambdaEij(i, j)))
                               + log(I * (scaleE * scaleE - exp(2 * lambdaEij(i, j))));
                    gLambdaEij(i,j) += - 1.0
                                    + 1.0 / (log(scaleE) - lambdaEij(i, j))
                                    - 2.0 * exp(2 * lambdaEij(i, j))
                                      / (scaleE * scaleE - exp(2 * lambdaEij(i, j)));
                }
            }
    } else if (options->hyperprior == PRIOR_EXPONENTIAL) {
        /* Laplacian prior: exponentially distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            fx += -log(2.0) - log(scaleH) - 2.0 * lambdaHi(i)
                       + scaleH * exp(2.0 * lambdaHi(i));
            gLambdaHi(i) += -2.0 + 2.0 * scaleH * exp(2.0 * lambdaHi(i));
        }
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                fx += -log(2.0) - log(scaleE) - 2.0 * lambdaEij(i,j)
                       + exp(2.0 * lambdaEij(i,j) + log(scaleE));
                gLambdaEij(i,j) += -2.0
                                + 2.0 * exp(2.0 * lambdaEij(i,j) + log(scaleE));
            }
    } else {
        /* The Horseshoe: Half-Cauchy distributed variances */
        for (int i = 0; i < ali->nSites; i++) {
            fx += -lambdaHi(i) + log(scaleH * scaleH + exp(2 * lambdaHi(i)))
                       -log(2 * scaleH / PI);
            gLambdaHi(i) += -1 + 2.0 * exp(2 * lambdaHi(i))
                      / (scaleH * scaleH + exp(2 * lambdaHi(i)));
        }
        for (int i = 0; i < ali->nSites-1; i++)
            for (int j = i + 1; j < ali->nSites; j++) {
                fx += -lambdaEij(i, j)
                           + log(scaleE * scaleE + exp(2 * lambdaEij(i, j)))
                           -log(2 * scaleE / PI);
                gLambdaEij(i,j) += -1 + 2.0 * exp(2 * lambdaEij(i, j))
                      / (scaleE * scaleE + exp(2 * lambdaEij(i, j)));
            }
    }
    return fx;
}

void ZeroAPCPriors(alignment_t *ali, options_t *options, numeric_t *lambdas,
    lbfgsfloatval_t *x) {
    /* Compute the variances of the couplings for each pair */
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++) {
            /* Mean(eij) over ai, aj */
            numeric_t mean = 0.0;
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    mean += xEij(i, j, ai, aj);
            mean *= 1.0 / ((numeric_t) ali->nCodes * ali->nCodes);

            /* Var(eij) over ai, aj */
            numeric_t ssq = 0.0;
            for (int ai = 0; ai < ali->nCodes; ai++)
                for (int aj = 0; aj < ali->nCodes; aj++)
                    ssq += (xEij(i, j, ai, aj) - mean)
                         * (xEij(i, j, ai, aj) - mean);
            /* Use N rather than N-1 since N has better MSE */
            numeric_t var = ssq / ((numeric_t) (ali->nCodes * ali->nCodes));
            lambdaEij(i, j) = var;
        }

    /* Determine the site-wise statistics of the variances */
    numeric_t nPairs =  ((numeric_t) ((ali->nSites) * (ali->nSites - 1))) / 2.0;
    numeric_t V_avg = 0.0;
    numeric_t *V_pos_avg = (numeric_t *) malloc(ali->nSites * sizeof(numeric_t));
    for (int i = 0; i < ali->nSites; i++) {
        V_pos_avg[i] = 0.0;
    }
    for (int i = 0; i < ali->nSites - 1; i++) {
        for (int j = i + 1; j < ali->nSites; j++) {
            V_pos_avg[i] += lambdaEij(i, j) / (numeric_t) (ali->nSites - 1);
            V_pos_avg[j] += lambdaEij(i, j) / (numeric_t) (ali->nSites - 1);
            V_avg += lambdaEij(i, j) / nPairs;
        }
    }

    /* Remove the first component of the variances */
    for (int i = 0; i < ali->nSites - 1; i++)
        for (int j = i + 1; j < ali->nSites; j++)
            lambdaEij(i, j) =
                lambdaEij(i, j) - V_pos_avg[i] * V_pos_avg[j] / V_avg;

    /* Transform and truncate variances into lambda hyperparameters */
    numeric_t pcount = 0.0;
    numeric_t psum = 0.0;
    numeric_t inbounds = 0;
    numeric_t min = LAMBDA_J_MAX;
    numeric_t max = LAMBDA_J_MIN;
    for (int i = 0; i < ali->nSites - 1; i++) {
        for (int j = i + 1; j < ali->nSites; j++) {
            /* Lambda coefficients are 1/2 the inverse variance */
            if (lambdaEij(i, j) > 0) {
                lambdaEij(i, j) = 1.0 / (2.0 * lambdaEij(i, j));
                psum += lambdaEij(i, j);
                pcount += 1.0;
            } else {
                lambdaEij(i, j) = LAMBDA_J_MAX + 1.0;
            }

            /* Truncate lambda for numerical stability */
            if (lambdaEij(i, j) >= LAMBDA_J_MIN && lambdaEij(i, j) <= LAMBDA_J_MAX)
                inbounds += 1.0 / (numeric_t) ((ali->nSites)*(ali->nSites - 1) / 2.0);
            if (lambdaEij(i, j) < 0 || !isfinite(lambdaEij(i, j)))
                lambdaEij(i, j) = LAMBDA_J_MAX;
            if (lambdaEij(i, j) < LAMBDA_J_MIN) lambdaEij(i, j) = LAMBDA_J_MIN;
            if (lambdaEij(i, j) > LAMBDA_J_MAX) lambdaEij(i, j) = LAMBDA_J_MAX;

            /* Track extremes */
            if (lambdaEij(i, j) > max) max = lambdaEij(i, j);
            if (lambdaEij(i, j) < min) min = lambdaEij(i, j);
        }
    }
    fprintf(stderr, "Raw coupling hyperparameter statistics:\n"
                    "\tMean positive lambda: %f\n"
                    "\tPercent of ij's positive: %f\n"
                    "\tPercent in bounds (%f < L < %f): %f\n",
                    psum / pcount,
                    pcount / nPairs,
                    min, max, inbounds);
}

const char *LBFGSErrorString(int ret) {
    const char *p;
    switch(ret) {
        case LBFGSERR_UNKNOWNERROR:
            p = "UNKNOWNERROR";
            break;
        /** Logic error. */
        case LBFGSERR_LOGICERROR:
            p = "LOGICERROR";
            break;
        /** Insufficient memory. */
        case LBFGSERR_OUTOFMEMORY:
            p = "OUTOFMEMORY";
            break;
        /** The minimization process has been canceled. */
        case LBFGSERR_CANCELED:
            p = "CANCELED";
            break;
        /** Invalid number of variables specified. */
        case LBFGSERR_INVALID_N:
            p = "INVALID_N";
            break;
        /** Invalid number of variables (for SSE) specified. */
        case LBFGSERR_INVALID_N_SSE:
            p = "INVALID_N_SSE";
            break;
        /** The array x must be aligned to 16 (for SSE). */
        case LBFGSERR_INVALID_X_SSE:
            p = "INVALID_X_SSE";
            break;
        /** Invalid parameter lbfgs_parameter_t::epsilon specified. */
        case LBFGSERR_INVALID_EPSILON:
            p = "INVALID_EPSILON";
            break;
        /** Invalid parameter lbfgs_parameter_t::past specified. */
        case LBFGSERR_INVALID_TESTPERIOD:
            p = "INVALID_TESTPERIOD";
            break;
        /** Invalid parameter lbfgs_parameter_t::delta specified. */
        case LBFGSERR_INVALID_DELTA:
            p = "INVALID_DELTA";
            break;
        /** Invalid parameter lbfgs_parameter_t::linesearch specified. */
        case LBFGSERR_INVALID_LINESEARCH:
            p = "INVALID_LINESEARCH";
            break;
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        case LBFGSERR_INVALID_MINSTEP:
            p = "INVALID_MINSTEP";
            break;
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        case LBFGSERR_INVALID_MAXSTEP:
            p = "INVALID_MAXSTEP";
            break;
        /** Invalid parameter lbfgs_parameter_t::ftol specified. */
        case LBFGSERR_INVALID_FTOL:
            p = "INVALID_FTOL";
            break;
        /** Invalid parameter lbfgs_parameter_t::wolfe specified. */
        case LBFGSERR_INVALID_WOLFE:
            p = "INVALID_WOLFE";
            break;
        /** Invalid parameter lbfgs_parameter_t::gtol specified. */
        case LBFGSERR_INVALID_GTOL:
            p = "INVALID_GTOL";
            break;
        /** Invalid parameter lbfgs_parameter_t::xtol specified. */
        case LBFGSERR_INVALID_XTOL:
            p = "INVALID_XTOL";
            break;
        /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
        case LBFGSERR_INVALID_MAXLINESEARCH:
            p = "INVALID_MAXLINESEARCH";
            break;
        /** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
        case LBFGSERR_INVALID_ORTHANTWISE:
            p = "INVALID_ORTHANTWISE";
            break;
        /** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
        case LBFGSERR_INVALID_ORTHANTWISE_START:
            p = "INVALID_ORTHANTWISE_START";
            break;
        /** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
        case LBFGSERR_INVALID_ORTHANTWISE_END:
            p = "ORTHANTWISE_END";
            break;
        /** The line-search step went out of the interval of uncertainty. */
        case LBFGSERR_OUTOFINTERVAL:
            p = "OUTOFINTERVAL";
            break;
        /** A logic error occurred; alternatively: the interval of uncertainty
            became too small. */
        case LBFGSERR_INCORRECT_TMINMAX:
            p = "INCORRECT_TMINMAX";
            break;
        /** A rounding error occurred; alternatively: no line-search step
            satisfies the sufficient decrease and curvature conditions. */
        case LBFGSERR_ROUNDING_ERROR:
            p = "ROUNDING_ERROR";
            break;
        /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
        case LBFGSERR_MINIMUMSTEP:
            p = "MINIMUMSTEP";
            break;
        /** The line-search step became larger than lbfgs_parameter_t::max_step. */
        case LBFGSERR_MAXIMUMSTEP:
            p = "MAXILBFGSERR_MUMSTEP";
            break;
        /** The line-search routine reaches the maximum number of evaluations. */
        case LBFGSERR_MAXIMUMLINESEARCH:
            p = "MAXIMUMLINESEARCH";
            break;
        /** The algorithm routine reaches the maximum number of iterations. */
        case LBFGSERR_MAXIMUMITERATION:
            p = "MAXIMUMITERATION";
            break;
        /** Relative width of the interval of uncertainty is at most
            lbfgs_parameter_t::xtol. */
        case LBFGSERR_WIDTHTOOSMALL:
            p = "WIDTHTOOSMALL";
            break;
        /** A logic error (negative line-search step) occurred. */
        case LBFGSERR_INVALIDPARAMETERS:
            p = "INVALIDPARAMETERS";
            break;
        /** The current search direction increases the objective function value. */
        case LBFGSERR_INCREASEGRADIENT:
            p = "INCREASEGRADIENT";
            break;
        case 0:
            p = "Minimization success";
            break;
        default:
            p = "No detected error";
            break;
    }
    return p;
}
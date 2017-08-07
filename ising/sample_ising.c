#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"

/*
 * SAMPLE_ISING.C
 *
 * Samples from an Ising system with Gibbs sampling.
 * 
 * Usage:
 *  [x] = sample_ising(h, J, x, n_steps);
 *
 * Compilation:
 *  mex -lm CFLAGS='-O3 -fPIC -std=c99' sample_ising.c
 *
 */

#define randu()         (double)rand() / (double)RAND_MAX

int rand_lim(int limit) {
    /* return a random integer between 0 and limit, inclusive.
     *      -from stackoverflow user Jerry Coffin-        */
    int divisor = RAND_MAX/(limit+1);
    int retval;
    do {
        retval = rand() / divisor;
    } while (retval > limit);
    return retval;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Interface to MATLAB */

    /* Input */
    double *h = mxGetPr(prhs[0]);         /* N x 1 */
    double *J = mxGetPr(prhs[1]);         /* N x N */
    double *x_init = mxGetPr(prhs[2]);    /* N x n_particles */
    int n_steps = (int) *mxGetPr(prhs[3]);

    /* Determine dimensions of system */
    int N = mxGetDimensions(prhs[0])[0];
    int n_particles = mxGetDimensions(prhs[2])[1];

    /* Output */
    plhs[0] = mxCreateDoubleMatrix(N, n_particles, mxREAL);
    double *x = (double *) mxGetPr(plhs[0]);
    
    /* Copy initialization of spins */
    for (int i = 0; i < n_particles * N; i++) x[i] = x_init[i];
    
    /* Advance chains */
    for (int i = 0; i < n_particles; i++)
        for (int s = 0; s < n_steps; s++) { 
            /* Select a random site */
            int ix = rand_lim(N - 1);
            
            /* Compute conditional distribution */
            double H = h[ix];
            for (int j = 0; j < N; j++)
                H += J[j + ix * N] * x[j + i * N];
            double P = exp(2 * H) / ( 1 + exp(2 * H));
            
            /* Flip spin accordingly */
            x[ix + i * N] = (randu() < P) * 2.0 - 1.0;
        }
}

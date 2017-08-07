#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"

/*
 * GIBBS_POTTS.C
 *
 * Heat-bath sampling of a system of Potts spins {A_i}, i=1...L
 * with a Hamiltonian specified by
 *      H(A_1, ... ,A_L) = -Sum_i h_i(A_i) -Sum_ij e_ij(A_i,A_j)
 *
 * Compilation:
 *  mex -lm CFLAGS='-O3 -fPIC -std=c99' gibbs_potts.c
 *
 * Usage:
 *  [sample, energies] = gibbs_potts(hi, eij, burnin, N)
 */

#define Hi(i,j)          hi[i + n_spins*j]
#define Eij(i,j,k,l)    eij[i + n_spins*(j + n_spins*(k + n_states*l))]
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

double hamiltonian(int n_states, int n_spins, int *sequence, double *hi,
    double *eij) {
    /* Compute energy for a configuration specified by SEQUENCE */
    double energy = 0;
    for (int i = 0; i < n_spins; i++) {
        /* field contributions */
        energy -= Hi(i, sequence[i]);
        /* coupling contributions */
        for (int j = i+1; j < n_spins; j++) {
            energy -= Eij(i, j, sequence[i], sequence[j]);
        }
    }
    return energy;
}

double delta_hamiltonian(int n_states, int n_spins, int *sequence, int site,
    int spin, double *hi, double *eij) {
    /* Compute change in energy when position SITE in the spin configuration
     * SEQUENCE is modified to SPIN */

    /* field change contribution */
    double energy = -Hi(site, spin) + Hi(site, sequence[site]);

    /* coupling change contribution */
    for (int j = 0; j < site; j++)
        energy -= Eij(site, j, spin, sequence[j])
               - Eij(site, j, sequence[site], sequence[j]);
    for (int j = site + 1; j < n_spins; j++)
        energy -= Eij(site, j, spin, sequence[j])
               - Eij(site, j, sequence[site], sequence[j]);
    return energy;
}


void sample(int n_steps, int n_states, int n_spins, int *seq, double *energy,
    double *hi, double *eij) {
    double *factors_cumsum = (double *) mxMalloc(n_states * sizeof(double));
    double *factors_dH = (double *) mxMalloc(n_states * sizeof(double));

    for (int i = 0; i < n_steps; i++) {
        /* Select a random site Ai */
        int site = rand_lim(n_spins-1);

        /* Thermalize according to the conditional distribution */
        double Z = 0;
        for (int j = 0; j < n_states; j++) {
            factors_dH[j] =
                delta_hamiltonian(n_states, n_spins, seq, site, j, hi, eij);
            Z += exp(-factors_dH[j]);
            factors_cumsum[j] = Z;
        }
        double p = randu() * Z;
        int spin = 0;
        while (p > factors_cumsum[spin]) spin++;
        seq[site] = spin;
        *energy += factors_dH[spin];
    }
    mxFree(factors_cumsum);
    mxFree(factors_dH);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* interface to MATLAB */

    /* Input */
    double *hi = mxGetPr(prhs[0]);          /* L x q */
    double *eij = mxGetPr(prhs[1]);         /* L x L x q x q */
    int burn_in = (int) *mxGetPr(prhs[2]);
    int n_steps = (int) *mxGetPr(prhs[3]);

    /* Determine dimensions of system */
    int n_spins = mxGetDimensions(prhs[0])[0];
    int n_states = mxGetDimensions(prhs[0])[1];
    mexPrintf("\nSampling parameters:\n %i spins x %i states \n"
              "N_STEPS = %i \n", n_spins, n_states, n_steps);

    /* Output */
    plhs[0] = mxCreateNumericMatrix(n_steps, n_spins, mxINT32_CLASS, mxREAL);
    int *samples = (int *) mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(n_steps, 1, mxREAL);
    double *energies = (double *) mxGetPr(plhs[1]);

    /* Generate random initial sequence */
    srand(time(0));
    int *seq = (int *) mxCalloc(n_spins, sizeof(int));
    for (int i = 0; i < n_spins; i++) seq[i] = rand_lim(n_states - 1);
    double energy = hamiltonian(n_states, n_spins, seq, hi, eij);

    /********************************* Burn-in ********************************/
    sample(burn_in, n_states, n_spins, seq, &energy, hi, eij);    

    /***************************** Gibbs sampling *****************************/
    int skip = n_spins * n_states;

    for (int i = 0; i < n_steps; i++) {
        sample(skip, n_states, n_spins, seq, &energy, hi, eij);

        /* Store every SKIP sequences */
        energies[i] = energy;
        for (int j = 0; j < n_spins; j++) samples[i + n_steps * j] = seq[j] + 1;
    }
}

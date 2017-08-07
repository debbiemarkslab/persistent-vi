# pvi

pvi implements Persistent Variational Inference for learning Potts models as well as regularized Maximum pseudolikelihood. 

## Usage
      pvi [options] alignmentfile
      pvi -c couplingsfile alignmentfile
      pvi -o paramfile -c couplingsfile alignmentfile
      pvi [-h | --help]
      
    Required input:
      alignmentfile                    Multiple sequence alignment in FASTA format

    Options, output:
      -c  --couplings  couplingsfile   Save coupling scores to file (text)
      -o  --output     paramfile       Save estimated parameters to file (binary)

    Options, alignment processing:
      -s  --scale      <value>         Sequence weights: neighborhood weight [s > 0]
      -t  --theta      <value>         Sequence weights: neighborhood divergence [0 < t < 1]

    Options, Maximum a posteriori estimation (L-BFGS, default):
      -lh --lambdah    <value>         Set L2 lambda for fields (h_i)
      -le --lambdae    <value>         Set L2 lambda for couplings (e_ij)

    Options, Bayesian estimation (MCMC):
      -v --variational                 Approximation posterior by Gaussian

    Options, general:
      -a  --alphabet   alphabet        Alternative character set to use for analysis
      -f  --focus      identifier      Select only uppercase, non-gapped sites from a focus sequence
      -g  --gapignore                  Model sequence likelihoods only by coding, non-gapped portions
      -i  --independent                Estimate a site-independent model
      -m  --maxiter                    Maximum number of iterations
      -n  --ncores    [<number>|max]   Maximum number of threads to use in OpenMP
      -h  --help                       Usage

## Compilation
pvi requires no external libraries, but optionally requires OpenMP to support multicore calculations

**Multicore**. To compile with `gcc` and OpenMP: 

    make all-openmp

**Recommended for Linux**. To compile with `gcc`: 

    make all

**Recommended for Mac OS X**. To compile with `clang`:

    make all-mac

**Single precision**. All of the above targets compile to double precision (64 bit), but reducing the precision to single (32 bit) increases speed and decreases memory requirements by approximately a factor of two. The fastest compile settings are:

    make all-openmp32

## Examples (pseudolikelihood)
**Standard protein alignment**. The following example command infers the parameters to a model of an alignment of the protein dihdyrofolate reductase (DHFR) with regularization parameters λ<sub>e</sub> = 1.0, λ<sub>h</sub> = 1.0 and the maximum number of iterations at 100:

    bin/pvi -o example/DHFR/DHFR.eij -f DYR_ECOLI -le 1.0 -lh 1.0 -m 100 example/DHFR/DHFR.a2m

**Reduced alphabet**. Although the default alphabet is "-ACDEFGHIKLMNPQRSTVWY", reduced systems can be encoded with arbitrary alphabets. As an example, simulated draws from a 3-state, 1-dimensional Potts model are provided in the examples folder and encoded by the characters _, *, and ^. The following command would estimate the parameters by running to convergence with λ<sub>e</sub> = 1.0, λ<sub>h</sub> = 1.0 and sequence reweighting disabled:

    bin/pvi -c example/potts/potts3.txt -a _*^ -t -1 -le 1.0 -lh 1.0 example/potts/potts3.a2m
A Potts model will only have interactions between i -> i + 1, and this can be inspected in the coupling scores in the output example/potts/potts3.txt 

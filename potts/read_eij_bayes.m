function params = read_eij_bayes(paramfile)
%READ_EIJ reads a binary file of parameters for a pairwise maximum entropy
%  model (undirected graphical model) estimated by plm. The model describes
%  the distribution of sequences of length L drawn from an alphabet with q
%  characters. The outputs are:
%   
%   Object      Description                         Dimensions
%   hi          sitewise fields     hi(i,Ai)        L x q
%   eij         pairwise couplings  eij(i,j,Ai,Aj)  L x L x q x q
%   fi          sitewise marginals  fi(i,Ai)        L x q
%   fij         pairwise marginals  fij(i,j,Ai,Aj)  L x L x q x q
%   meta        metadata
%
%   Note that both the eij and fij arrays are output in dense form, but 
%   will be symmetric under (i,j,ai,aj) <-> (j,i,aj,ai)
%

PRECISION = 'single';

f_eij = fopen(paramfile, 'r');
%
L = fread(f_eij, 1, 'int');
q = fread(f_eij, 1, 'int');
params.L = L;
params.q = q;
params.target_seq = char(fread(f_eij, L, 'char'))';
params.offset_map = fread(f_eij, L, 'int');
%
params.logsigi_mu = fread(f_eij, L, PRECISION)';
params.logsigi_std = fread(f_eij, L, PRECISION)';
params.logsigij_mu = squareform(fread(f_eij, L * (L-1)/2, PRECISION));
params.logsigij_std = squareform(fread(f_eij, L * (L-1)/2, PRECISION));
%
params.fi = fread(f_eij, [q L], PRECISION)';
params.hi_mu = fread(f_eij, [q L], PRECISION)';
params.hi_std = fread(f_eij, [q L], PRECISION)';
%
params.fij = zeros(L, L, q, q);
params.eij_mu = zeros(L, L, q, q);
params.eij_std = zeros(L, L, q, q);
for i=1:(L-1)
    for j=(i+1):L
        ij = fread(f_eij, [2 1], 'int');
        params.fij(i,j,:,:) = fread(f_eij, [q q], PRECISION)';
        params.fij(j,i,:,:) = squeeze(params.fij(i,j,:,:))';
        params.eij_mu(i,j,:,:) = squeeze(fread(f_eij, [q q], PRECISION))';
        params.eij_mu(j,i,:,:) = squeeze(params.eij_mu(i,j,:,:))';
        params.eij_std(i,j,:,:) = squeeze(fread(f_eij, [q q], PRECISION))';
        params.eij_std(j,i,:,:) = squeeze(params.eij_std(i,j,:,:))';
    end
end

fclose(f_eij);
end
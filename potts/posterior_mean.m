function [ hi, eij ] = posterior_mean(params)
%POSTERIOR_MEAN Computes the posterior mean of parameter estimates under
% the variationally inferred hierarchical Boltzmann Relevance Machine
%
L = params.L;
q = params.q;
hi  = zeros(size(params.hi_mu));
eij = zeros(size(params.eij_mu));
for i = 1:L
%     sigma = exp(params.logsigi_mu(i));
    sigma = exp(0.5 * params.logsigi_std(i)^2 + params.logsigi_mu(i));
    hi(i,:) = sigma * params.hi_mu(i,:);
end

for i = 1:(L-1)
    for j = (i+1):L
%         sigma = exp(params.logsigij_mu(i,j));
        sigma = exp(0.5 * params.logsigij_std(i,j)^2 + params.logsigij_mu(i,j));
        E = sigma * params.eij_mu(i,j,:,:);
        % Shift to 0-sum gauge
        % bsxfun(@minus,bsxfun(@minus, E, mean(E,2)),mean(E,1)) + mean(E(:));
        eij(i,j,:,:) = E;
        eij(j,i,:,:) = squeeze(eij(i,j,:,:))';
    end
end
end


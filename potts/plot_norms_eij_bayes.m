function FN = plot_norms_eij_bayes(params, slice)
%COMPUTE_NORMS
%
eij = params.eij_mu;
if(nargin < 2)
    slice = 1:size(eij,4);
end
%
N = size(eij,1);
FN = zeros(N,N);
%
% Residue
%
for i=1:(N-1)
    for j=i+1:N
        % This will be distributed as the product of a lognormal & a
        % chi-distribution
        sigma = exp(0.5 * params.logsigij_std(i,j)^2 + params.logsigij_mu(i,j));
        E = sigma * squeeze(eij(i,j,slice,slice));

        % Convert to 0-sum Gauge
        E = bsxfun(@minus,bsxfun(@minus, E, mean(E,2)),mean(E,1)) + mean(E(:));
        FN(i,j) = norm(E,'fro');
        FN(j,i) = FN(i,j);
    end
end
if nargout == 0
    imagesc(FN)
    caxis([-1 1] * max(FN(:)));
    axis square
    title('Coupling Strength (Norm)');
    colormap(cbrewer('div', 'RdBu', 128));
    set(gcf,'color','w')
end
end
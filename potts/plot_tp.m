function plot_tp(C, S, color)
%PLOT_TP [lots the true positive rate for a binary contact map
%
C = C > 0;
%
L = size(C,1);
[Ix, Jx] = ndgrid(1:L, 1:L);
C_true = C(Ix > Jx);
[~, ix] = sort(S(Ix > Jx),'descend');
if nargin == 3
    plot(cumsum(C_true(ix))' ./ (1:numel(C_true)),...
        'color',color,'linewidth',1)
else
    plot(cumsum(C_true(ix))' ./ (1:numel(C_true)),...
        'linewidth',2)
end
end
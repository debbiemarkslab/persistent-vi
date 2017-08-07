function plot_tp(C, S, d)
%PLOT_TP [lots the true positive rate for a binary contact map
%
C = C > 0;
%
L = size(C,1);
[Ix, Jx] = ndgrid(1:L, 1:L);
C_true = C(Ix > Jx + d);
[~, ix] = sort(S(Ix > Jx + d),'descend');
plot(cumsum(C_true(ix))' ./ (1:numel(C_true)))
end
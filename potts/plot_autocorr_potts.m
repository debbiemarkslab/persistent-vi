function plot_autocorr_potts(sample)
%POTTS_AUTOCORR plots the autocorrelation for samples from a Potts model
%
numLags = 30;
hold on
q = max(sample(:));
ACF = zeros(size(sample, 2), q, numLags + 1);
for i=1:size(sample, 2)
    for a = 1:q
        ACF(i,a,:) = autocorr(sample(:,i) == a, numLags);
        plot(squeeze(ACF(i,a,:)));
    end
end    
hold off
end

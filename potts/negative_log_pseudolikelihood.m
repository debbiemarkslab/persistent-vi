function nlp = negative_log_pseudolikelihood(sample, hi, eij)
%NEGATIVE_LOG_PSEUDOLIKELIHOOD Computes NLP given Potts(hi, eij)
%
nlp = 0;
L = size(hi,1);
for s = 1:size(sample,1)
    for i = 1:L
        H = hi(i,:)';
        for j = 1:L
            if i ~= j
                H = H + squeeze(eij(i, j, :, sample(s, j)));
            end
        end
        % H = exp(H) / sum(exp(H));
        % nlp = nlp - log(H(sample(s,i)));
        nlp = nlp - log(exp(H(sample(s,i))) / sum(exp(H)));
    end
end
end


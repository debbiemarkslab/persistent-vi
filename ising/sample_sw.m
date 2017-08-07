function X = sample_sw(h, J, warmup, iter)
% SAMPLE_SW Swendsen-Wang sampling for Ising models on {-1, 1}
%
% John Ingraham, 2016

% Initialize state
L = numel(h);
X = zeros(iter, L);
x = ones(L, 1);
x = 2 * randi(2, L, 1) - 3;

% Precompute bond conditional probabilities
P = 1 - exp(-2 * abs(squareform(J)));

% Output
fprintf(['Sampling\n']);
for t = 1:(warmup + iter)
    % Report every 1000 steps
    if mod(t - warmup, 1000) == 0
        fprintf(['Sampling: ' num2str(t - warmup) ...
                 ' of ' num2str(iter) '\n']);
    end
    
    % Sample bonds given spins
    valid = squareform(J .* (x * x') > 0);
    B = sparse(squareform(valid .* (rand(size(P)) < P)));

    % Sample spins given bonds (Matlab does the DFS for percolation)
    [~, C] = graphconncomp(B, 'Directed', 'false');
    for i = 1:max(C)
        % Flip with probability determined by bias
        z = rand < 1 / (1 + exp(-2 * sum(h(C == i) .* x(C == i))));
        x(C == i) = (2 * z - 1) * x(C == i);
    end

    % Store with probability p
    if (t > warmup)
        X(t - warmup,:) = x(:);
    end
end
end


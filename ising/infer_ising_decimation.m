function [params loss] = infer_ising_decimation(data, varargin)
%TRAIN_ISING trains an Ising model for data using L1-regularized
%  pseudolikelihood or Minimum Probability Flow
%
% Based on 

% Default parameters
num_iterations = 1000;
lambda_h = 0;
lambda_J = 0;
lambda_l1 = 0;
cross_validate = 0;
display_mode = 'none';

% Initialize site biases h and pairwise couplings J
N = size(data, 2);
h = zeros(N,1);
J = zeros(N,N);

% Parse arguments
for kx = 1:2:length(varargin)
    switch varargin{kx}
        case {'cross_validate'}
            cross_validate = varargin{kx+1};
        case {'num_iterations'}
            num_iterations = varargin{kx+1};
        case {'lambda_h'}
            lambda_h = varargin{kx+1};
        case {'lambda_J'}
            lambda_J = varargin{kx+1};
        case {'lambda_l1'}
            lambda_l1 = varargin{kx+1};
        case {'h_init'}
           h = varargin{kx+1};
        case {'J_init'}
           J = varargin{kx+1};
           % Enforce Jii = 0
           J = J - diag(diag(J));
        case {'display'}
           display_mode = varargin{kx+1};
    end
end

% Compute data-dependent statistics
% Encode data as spins [-1, +1]
up_ix = (data >= (max(data(:)) + min(data(:))) / 2);
data(up_ix) = 1;
data(~up_ix) = -1;

% Hyperparmeters
lambda.h = lambda_h;
lambda.J = lambda_J;
lambda.l1 = 0;
lambda.l1_eps = 1E-6;

% Initial decimation mask
decimate_frac = 0.05;
J_mask = squareform(ones(size(J)) - diag(diag(ones(size(J)))));

% Initial run
theta_init = pack_params(h, J);
gradfun = @(theta) ising_gradPL(theta, data, N, lambda, J_mask);
opts.Method = 'lbfgs';
opts.MaxIter = 100;
[theta, pl_max] = minFunc(gradfun, theta_init, opts);
[h, J] = unpack_params(theta, N);
new_tplf = 0;
tplf = 0;

while new_tplf >= tplf
    % Optimize the objective
    theta_init = pack_params(h, J);
    gradfun = @(theta) ising_gradPL(theta, data, N, lambda, J_mask);
    opts.Method = 'lbfgs';
    opts.MaxIter = 100;
    [theta, pl] = minFunc(gradfun, theta_init, opts);
    [h, J] = unpack_params(theta, N);
    % Decimate
    % Find
    J_flat = squareform(J);
    [~, ix] = sort(abs(J_flat),'ascend');
    % Only consider undecimated positions
    ix = setdiff(ix, find(J_mask == 0));
    % Resort the undecimated positions
    [~, ix_ix] = sort(abs(J_flat(ix)),'ascend');
    ix = ix(ix_ix);
    % Decimate DECIMATE_FRAC of the remaining couplings
    decimate_ix = ix(1:floor(end * decimate_frac));
    J_mask(decimate_ix) = 0;
    % Is the tilted PLF still improving?
    x = mean(J_mask);
    tplf = new_tplf;
    new_tplf = -pl + x * pl_max + (1 - x) * N * log(2);
    if new_tplf > tplf
        J_flat(decimate_ix) = 0;
        J = squareform(J_flat);
    end
    [new_tplf tplf]
end

% Output parameter structure
params.h = h;
params.J = J;
end

function [f, gradtheta] = ising_gradPL(theta, data, N, lambda, J_mask)
% Compute the negative log pseudolikelihood and gradient for an Ising model
%
    [h, J] = unpack_params(theta, N);
    gradh = zeros(size(h));
    gradJ = zeros(size(J));
    
    % Compute pseudolikelihood
    f = 0;
    for i = 1:size(data,1)
        % Conditional likelihood
        P = 1./ (1 + exp(-2 * data(i,:)' .* ...
                    (h + sum(bsxfun(@times, data(i,:), J), 2))));
        f = f - sum(log(P));
        gradh = gradh - 2*(1-P) .* data(i,:)';
        gJ = bsxfun(@times, 2*(1-P), data(i,:)' * data(i,:));
        gradJ = gradJ - gJ - gJ';
    end
    
    % Scale by the size of the data
    invN = 1 / size(data, 1);
    f = f * invN;
    gradh = gradh * invN;
    gradJ = gradJ * invN;

    % Add prior terms to form gradLogP, grad(+logPosterior)
    f = f + lambda.h * sum(h.^2) ...
          + lambda.J * sum(squareform(J).^2);
    % L2 regularization
    gradh = gradh + 2 * lambda.h * h;
    gradJ = gradJ + 2 * lambda.J * J;
    gradJ = gradJ - diag(diag(gradJ));

    % Mask only the unselected parameters
    gradJ = squareform(J_mask).* gradJ;

    % Pack negative gradient, grad(-logPosterior)
    gradtheta = pack_params(gradh, gradJ);
    
    % Callback function
    %callback(theta, N);
end

function theta = pack_params(h, J)
    % Reshape from parameters to vector
    theta = [h(:); squareform(J)'];
end

function [h, J] = unpack_params(theta, N)
    % Reshape from vector to parameters
    h = reshape(theta(1:N), [N 1]);
    J = squareform(theta(N + [1:N*(N-1)/2]));
end

function callback(theta, N)
% Optionally plot current state during inference
    [h, J] = unpack_params(theta, N);

    % Plot couplings
    figure(1)
    clf
    imagesc(J)
    colormap(blu_map);
    caxis(max(abs(J(:))) * [-1 1]);
    colorbar
    axis square
    drawnow
end
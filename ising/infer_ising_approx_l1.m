function [params loss] = infer_ising_approx_l1(data, varargin)
%TRAIN_ISING trains an Ising model for data using L1-regularized
%  pseudolikelihood or Minimum Probability Flow

% Default parameters
num_iterations = 1000;
lambda_h = 0;
lambda_J = 0;
lambda_l1 = 0;
cross_validate = 0;
method = 'pl';
display_mode = 'none';

% Initialize site biases h and pairwise couplings J
N = size(data, 2);
h = zeros(N,1);
J = zeros(N,N);

% Parse arguments
for kx = 1:2:length(varargin)
    switch varargin{kx}
        case {'method'}
            method = varargin{kx+1};
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

% Optimization function
optimizer = @L1General2_PSSgb;

% Optional: Compute a cross-validated loss
if cross_validate > 0
    theta_init = pack_params(h, J);
    cv_loss = zeros(cross_validate, 1);
    % Split data CROSS_VALIDATE ways
    ix = mod(randperm(size(data, 1)), cross_validate) + 1;
    for i = 1:cross_validate
        % Split data
        data_train = data(ix ~= i,:);
        data_test = data(ix == i,:);
        % Choose methods
        if strcmp(method, 'mpf')
            gradfun = @(theta) ising_gradMPF(theta, data_train, N, lambda);
            lossfun = @(theta) ising_lossMPF(theta, data_test, N);
        elseif strcmp(method, 'pl')
            gradfun = @(theta) ising_gradPL(theta, data_train, N, lambda);
            lossfun = @(theta) ising_lossPL(theta, data_test, N);
        end
        theta = optimizer(gradfun, theta_init, ...
                          lambda_l1 * ones(size(theta_init)));
        % Compute loss
        cv_loss(i) = lossfun(theta);
        [L, ~] = gradfun(theta);
        fprintf('Fold %d, test: %f train: %f \n', i, cv_loss(i), L);
    end
    loss = mean(cv_loss);
end

% Optimize the objective
theta_init = pack_params(h, J);
if strcmp(method, 'mpf')
    gradfun = @(theta) ising_gradMPF(theta, data, N, lambda);
elseif strcmp(method, 'pl')
    gradfun = @(theta) ising_gradPL(theta, data, N, lambda);
end
theta = optimizer(gradfun, theta_init, lambda_l1 * ones(size(theta_init)));
[h, J] = unpack_params(theta, N);

% Output parameter structure
params.h = h;
params.J = J;
end

function [f, gradtheta] = ising_gradPL(theta, data, N, lambda)
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

    % Pack negative gradient, grad(-logPosterior)
    gradtheta = pack_params(gradh, gradJ);
    
    % Callback function
    % callback(theta, N);
end

function f = ising_lossPL(theta, data, N)
% Compute the negative log pseudolikelihood for an Ising model
%
    [h, J] = unpack_params(theta, N);
    f = 0;
    for i = 1:size(data,1)
        % Conditional likelihood
        P = 1./ (1 + exp(-2 * data(i,:)' .* ...
                    (h + sum(bsxfun(@times, data(i,:), J), 2))));
        f = f - sum(log(P));
    end
    f = f / size(data,1);
end

function [f, gradtheta] = ising_gradMPF(theta, data, N, lambda)
% Compute the negative log pseudolikelihood and gradient for an Ising model
%
    [h, J] = unpack_params(theta, N);
    gradh = zeros(size(h));
    gradJ = zeros(size(J));
    
    % Compute MPF objective function
    f = 0;
    for i = 1:size(data,1)
        % MPF objective
        K = exp(-data(i,:)' .* (h + sum(bsxfun(@times, data(i,:), J), 2)) ...
                -log(size(data,1)));
        f = f + sum(K);
        gradh = gradh - data(i,:)' .* K;
        gJ = bsxfun(@times, -K, data(i,:)' * data(i,:));
        gradJ = gradJ +  gJ + gJ';
    end
    
    % Add prior terms to form gradLogP, grad(+logPosterior)
    f = f + lambda.h * sum(h.^2) ...
          + lambda.J * sum(squareform(J).^2);
    % L2 regularization
    gradh = gradh + 2 * lambda.h * h;
    gradJ = gradJ + 2 * lambda.J * J;
    gradJ = gradJ - diag(diag(gradJ));
    
    % Pack gradient
    gradtheta = pack_params(gradh, gradJ);

    % Callback function for debug
    % callback(theta, N);
end

function f = ising_lossMPF(theta, data, N)
% Compute the MPF objective and gradient for an Ising model
%
    [h, J] = unpack_params(theta, N);
    f = 0;
    for i = 1:size(data,1)
        % MPF objective
        K = exp(-data(i,:)' .* (h + sum(bsxfun(@times, data(i,:), J), 2)) ...
                -log(size(data,1)));
        f = f + sum(K);
    end
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
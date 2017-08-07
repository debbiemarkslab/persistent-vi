function params = infer_ising_pcd(data, varargin)
%TRAIN_ISING trains an Ising model for data using Boltzmann learning
%   with persistent Markov Chains

% Default parameters
num_iterations = 1000;
num_particles = 100;
num_gsweeps = 1;
lambda_h = 0;
lambda_J = 0;
lambda_l1 = 0;
l1_eps = 1E-8;
dropout = 0;
burnin = 'random';

% learn_params.method = 'SGD';
% learn_params.alpha = 1E-2;

% Adam learning with annealed learning rate and momentum near the end
learn_params.method = 'AdamAnneal';
learn_params.alpha = 1E-4;
learn_params.beta1 = 0.99;
learn_params.beta2 = 0.999;

% Adam learning
% learn_params.method = 'Adam';
% learn_params.alpha = 1E-4;
% learn_params.beta1 = 0.99;
% learn_params.beta2 = 0.999;
plot_mode = 'off';

% Initialize site biases h and pairwise couplings J
N = size(data, 2);
h = zeros(N,1);
J = zeros(N,N);

% Parse arguments
for kx = 1:2:length(varargin)
    switch varargin{kx}
        case {'num_iterations'}
            num_iterations = varargin{kx+1};
        case {'num_particles'}
            num_particles = varargin{kx+1};
        case {'num_gsweeps'}
            num_gsweeps = varargin{kx+1};
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
        case {'burnin'}
           burnin = varargin{kx+1};
        case {'optopts'}
           learn_params = varargin{kx+1};
        case {'plot'}
           plot_mode = varargin{kx+1};
    end
end

% Compute data-dependent statistics
% Encode data as spins [-1, +1]
up_ix = (data >= (max(data(:)) + min(data(:))) / 2);
data(up_ix) = 1;
data(~up_ix) = -1;
fi = mean(data)';
fij = cov(data) + fi * fi';

% Initialize Markov Chains
num_gsteps = num_gsweeps * N;

% Random initialization
x = 2 * (randi(2, num_particles, N) - 1) - 1;
switch burnin
    case 'long'
        x = sample_ising(h, J, x', 1000); x = x';
    case 'data'
        x = data(randsample(size(data,1),num_particles,'true'),:);
end

% Optimize the objective
theta_init = pack_params(h, J);
gradfun = @(theta, i) ising_gradNLP(theta);
theta = optimize_sgd(gradfun, theta_init, num_iterations, learn_params, ...
                    @callback);
[h, J] = unpack_params(theta, N);

% Output parameter structure
params.h = h;
params.J = J;

function gradtheta = ising_gradNLP(theta)
% Compute a noisy estimate of the gradient using Persistent Markov Chains
    % Compute the gradient of the log likelihood an Ising model
    [h, J] = unpack_params(theta, N);
    
    % Optional dropout
    if dropout
        h_mask = randi(2, size(h)) - 1;
        J_mask = squareform(randi(2, size(squareform(J))) - 1);
        h = 2 * h .* h_mask;
        J = 2 * J .* J_mask;
    end
    
    % Sample from current parameters
    x = x';
    x = sample_ising(h, J, x, num_gsteps);
    x = x';
    
    % Compute the equilibrium statistics, grad(+logLikelihood)
    fi_model = mean(x)';
    fij_model = cov(x) + fi_model * fi_model';
    gradh = fi - fi_model;
    gradJ = fij - fij_model;
    gradJ = gradJ - diag(diag(gradJ));

    % Add prior terms to form gradLogP, grad(+logPosterior)
    gradh = gradh - 2 * lambda_h * h;
    gradJ = gradJ - 2 * lambda_J * J;
    % L1 regularization
    gradh = gradh - lambda_l1 * h ./ sqrt(h.^2 + l1_eps);
    gradJ = gradJ - lambda_l1 * J ./ sqrt(J.^2 + l1_eps);
    
    % Optional dropout
    if dropout
        gradh = 2 * gradh .* h_mask;
        gradJ = 2 * gradJ .* J_mask;
    end
    
    % Pack negative gradient, grad(-logPosterior)
    gradtheta = pack_params(-gradh, -gradJ);
end

function callback(theta, ~, iter)
% Optionally plot current state during inference
    if ~strcmp(plot_mode, 'off') && mod(iter, 100) == 0
        [h, J] = unpack_params(theta, N);
        
        % Plot couplings
        figure(1)
        clf
        imagesc(J)
        colormap(blu_map);
        caxis(max(abs(J(:))) * [-1 1])
        colorbar
        axis square

        if strcmp(plot_mode, 'full')
            % Plot current state of Markov chains
            figure(2)
            clf
            subplot(1,2,1)
            plot_mosaic(x, floor(sqrt(num_particles)), ...
                        floor(sqrt(N)),floor(sqrt(N)))

            % Plot data statistics versus model statistics
            subplot(1,2,2)
            hold on
            scatter(reshape(cov(data) + mean(data)' * mean(data), [N*N 1]), ...
                    reshape(cov(x) + mean(x)' * mean(x), [N*N 1]), '.')
            plot([-1 1],[-1 1],'k')
            xlabel('$\langle \sigma_i \sigma_j \rangle_\textrm{Data}$', ...
                'Interpreter', 'LaTeX')
            ylabel('$\langle \sigma_i \sigma_j \rangle_\textrm{Samples}$', ...
                'Interpreter', 'LaTeX')
            axis square
            xlim([-1 1])
            ylim([-1 1])
        end
        drawnow
    end
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
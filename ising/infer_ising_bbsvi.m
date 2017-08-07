function params = infer_ising_bbsvi(data, varargin)
%TRAIN_ISING trains an Ising model for data using Boltzmann learning
%   with persistent Markov Chains
%
% (c) John Ingraham, 2017

% Default parameters
num_iterations = 1000;
num_particles = 100;     % Number of persistent Markov chains
num_gsweeps = 1;         % Number of Gibbs sweeps (as in k of PCD-k)
num_samples = 1;         % Number of samples for SVI
var_h = 1.0;             % Gaussian prior variance, h
var_J = 1.0;             % Gaussian prior variance, J
scale_laplacian = 0;     % Laplacian prior scale parameter
laplacian_eps = 1E-8;
burnin = 'random';

% Adam learning with annealed learning rate and momentum near the end
learn_params.method = 'AdamAnneal';
learn_params.alpha = 1E-2;
learn_params.beta1 = 0.99;
learn_params.beta2 = 0.999;
plot_mode = 'off';

% Initialize site biases h and pairwise couplings J with a spike
N = size(data, 2);
mu_h = zeros(N,1);
mu_J = zeros(N,N);
logsig_h = -3 * ones(N,1);
logsig_J = -3 * (ones(N,N) - diag(diag(ones(N,N))));

% Parse arguments
for kx = 1:2:length(varargin)
    switch varargin{kx}
        case {'num_iterations'}
            num_iterations = varargin{kx+1};
        case {'num_particles'}
            num_particles = varargin{kx+1};
        case {'num_gsweeps'}
            num_gsweeps = varargin{kx+1};
        case {'num_samples'}
            num_samples = varargin{kx+1};
        case {'var_h'}
            var_h = varargin{kx+1};
        case {'var_J'}
            var_J = varargin{kx+1};
        case {'scale_laplacian'}
            scale_laplacian = varargin{kx+1};
        case {'h_init'}
           mu_h = varargin{kx+1};
        case {'J_init'}
           mu_J = varargin{kx+1};
           % Enforce Jii = 0
           mu_J = mu_J - diag(diag(mu_J));
        case {'burnin'}
           burnin = varargin{kx+1};
        case {'optopts'}
           learn_params = varargin{kx+1};
        case {'plot'}
           plot_mode = varargin{kx+1};
    end
end

% Encode data as spins [-1, +1]
up_ix = (data >= (max(data(:)) + min(data(:))) / 2);
data(up_ix) = 1;
data(~up_ix) = -1;

% Compute data-dependent statistics
fi = mean(data)';
fij = cov(data) + fi * fi';

% Initialize Markov Chains
num_gsteps = num_gsweeps * N;
% Random initialization
x = 2 * (randi(2, num_particles, N) - 1) - 1;
switch burnin
    case 'long'
        x = sample_ising(mu_h, mu_J, x', 1000); x = x';
    case 'data'
        x = data(randsample(size(data,1),num_particles,'true'),:);
end

% Optimize the objective
theta_init = [pack_params(mu_h, mu_J); pack_params(logsig_h, logsig_J)];
gradfun = @(theta, i) grad_ELBO(theta, @ising_gradNLP, num_samples);
theta = optimize_sgd(gradfun, theta_init, num_iterations, learn_params, ...
    @callback);

% Output variational parameters in structure
[mu_h, mu_J] = unpack_params(theta(1:end/2), N);
[logsig_h, logsig_J] = unpack_params(theta(end/2 + [1:end/2]), N);

% Posterior means and standard deviations
params.mu_h = mu_h;
params.mu_J = mu_J;
params.sig_h = transform_sigma(logsig_h);
params.sig_J = transform_sigma(logsig_J);

function gradtheta = ising_gradNLP(params)
% Compute a noisy estimate of the gradient of an Ising model using 
%  Persistent Markov Chains
    % Compute the gradient of the log likelihood an Ising model
    [h, J] = unpack_params(params, N);
    J = J - diag(diag(J));
    x = x';
    x = sample_ising(h, J, x, num_gsteps);
    x = x';

    % Compute the equilibrium statistics, grad(+logLikelihood)
    fi_model = mean(x)';
    fij_model = cov(x) + fi_model * fi_model';
    gradh = fi - fi_model;
    gradJ = fij - fij_model;
    gradJ = gradJ - diag(diag(gradJ));
    
    % Unnormalize the gradient
    gradh = size(data, 1) * gradh;
    gradJ = size(data, 1) * gradJ;

    % Add prior terms to form gradLogP, grad(+logPosterior)
    if var_h > 0
        gradh = gradh - h / var_h;
    end
    if var_J > 0
        gradJ = gradJ - J / var_J;
    end
    if scale_laplacian > 0
        gradh = gradh - h ./ (scale_laplacian * sqrt(h.^2 + laplacian_eps));
        gradJ = gradJ - J ./ (scale_laplacian * sqrt(J.^2 + laplacian_eps));
    end
    
    % Pack negative gradient, grad(-logPosterior)
    gradtheta = pack_params(-gradh, -gradJ);
end

function callback(theta, ~, iter)
% Optionally plot current state during inference
    if ~strcmp(plot_mode, 'off') && mod(iter, 100) == 0
        [h, J] = unpack_params(theta(1:end/2), N);
        [logsigh, logsigJ] = unpack_params(theta(end/2 +[1:end/2]), N);
        
        % Plot couplings
        figure(1)
        clf
        subplot(1,3,1)
        imagesc(J)
        colormap(blu_map);
        caxis(max(abs(J(:))) * [-1 1])
        colorbar
        axis square
        title('Coupling means')
        
        subplot(1,3,2)
        stdJ = exp(logsigJ);
        stdJ = stdJ - diag(diag(stdJ));
        imagesc(stdJ)
        caxis(max(stdJ(:)) * [-1 1])
        colorbar
        axis square
        title('Coupling std. deviations')
        
        subplot(1,3,3)
        hold on
        errorbar(h, 2 * exp(logsigh), 'bx')
        plot([1 numel(h)], [0 0], 'k');
        hold off
        ylim([-1 1] * max([1 max(abs(h))]));
        xlim([0 numel(h) + 1]);
        title('Fields 95% bounds')
        

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
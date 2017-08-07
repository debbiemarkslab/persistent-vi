function params = infer_ising_fadeout(data, varargin)
%TRAIN_ISING trains an Ising model for data using Boltzmann learning
%   with persistent Markov Chains
%
% (c) John Ingraham, 2017

% Default parameters
num_iterations = 1000;
num_particles = 64;      % Number of persistent Markov chains
num_gsweeps = 3;         % Number of Gibbs sweeps (as in k of PCD-k)
num_samples = 1;         % Number of samples for SVI
burnin = 'random';
hyperprior = 'horseshoe';

% Adam learning with annealed learning rate and momentum near the end
learn_params.method = 'AdamAnneal';
learn_params.alpha = 1E-2;
learn_params.beta1 = 0.99;
learn_params.beta2 = 0.999;
plot_mode = 'off';

% Initialize site biases h and pairwise couplings J with a spike
N = size(data, 2);
num_global = 0;
dZd = ones(N,N) - diag(diag(ones(N,N)));

% Initialize x at zero
mu_nch = zeros(N,1);
mu_ncJ = zeros(N,N);
logsig_nch = -3 * ones(N,1);
logsig_ncJ = -3 * dZd;

% Moment-matched to dropout
mu_logzh = -1 * ones(N,1);
mu_logzJ = -1 * dZd;
logsig_logzh = log(0.8) * ones(N,1);
logsig_logzJ = log(0.8) * dZd;

% Parse arguments
for kx = 1:2:length(varargin)
    switch varargin{kx}
        case {'num_iterations'}
            num_iterations = varargin{kx+1};
        case {'num_particles'}
            num_particles = varargin{kx+1};
        case {'num_samples'}
            num_samples = varargin{kx+1};
        case {'num_gsweeps'}
            num_gsweeps = varargin{kx+1};
        case {'hyperprior'}
            hyperprior = varargin{kx+1};
        case {'h_init'}
           mu_nch = varargin{kx+1};
        case {'J_init'}
           mu_ncJ = varargin{kx+1};
           % Enforce Jii = 0
           mu_ncJ = mu_ncJ - diag(diag(mu_ncJ));
        case {'burnin'}
           burnin = varargin{kx+1};
        case {'optopts'}
           learn_params = varargin{kx+1};
        case {'plot'}
           plot_mode = varargin{kx+1};
    end
end

% Initialize global hyperparameters (originally used N(-4,sqrt(3)))
if strcmp(hyperprior, 'lognormal')
    num_global = 4;
    mu_globals = [1 1 log(sqrt(3)) log(sqrt(3))];
    logsig_globals = -3 * ones(num_global, 1);
elseif strcmp(hyperprior, 'horseshoe')
    num_global = 2;
    mu_globals = -3 * [1 1];
    logsig_globals = -3 * ones(num_global, 1);
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
        x = sample_ising(mu_nch .* exp(mu_logzh),...
                         mu_ncJ .* exp(mu_logzJ), x', 1000); x = x';
    case 'data'
        x = data(randsample(size(data,1),num_particles,'true'),:);
end

% Optimize the objective
theta_init = [pack_params(mu_globals, mu_nch, mu_ncJ, mu_logzh, mu_logzJ); ...
              pack_params(logsig_globals, logsig_nch, logsig_ncJ, ...
                          logsig_logzh, logsig_logzJ)];
gradfun = @(theta, i) grad_ELBO(theta, ...
    @ising_gradNLP_noncentered, num_samples);
theta = optimize_sgd(gradfun, theta_init, num_iterations, ...
                    learn_params, @callback);

% Output variational parameters in structure
[mu_hyp, mu_nch, mu_ncJ, mu_logzh, mu_logzJ] = ...
    unpack_params(theta(1:end/2), N, num_global);
[logsig_hyp, logsig_nch, logsig_ncJ, logsig_logzh, logsig_logzJ] = ...
    unpack_params(theta(end/2 + [1:end/2]), N, num_global);

% Posterior means
params.mu_hyp = mu_hyp;
params.mu_nch = mu_nch;
params.mu_ncJ = mu_ncJ;
params.mu_logzh = mu_logzh;
params.mu_logzJ = mu_logzJ;

% Posterior standard deviations
params.sig_hyp = transform_sigma(logsig_hyp);
params.sig_nch = transform_sigma(logsig_nch);
params.sig_ncJ = transform_sigma(logsig_ncJ);
params.sig_logzh = transform_sigma(logsig_logzh);
params.sig_logzJ = transform_sigma(logsig_logzJ);

function gradtheta = ising_gradNLP_noncentered(params)
% Compute a noisy estimate of the gradient of an Ising model using 
%  Persistent Markov Chains
    % Transform from non-centered to centered parameters
    [hyper, nch, ncJ, logzh, logzJ] = unpack_params(params, N, num_global);
    h = nch .* exp(logzh);
    J = ncJ .* exp(logzJ);
    
    % Compute the centered gradient of the log likelihood an Ising model
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
    
    % Transform to a noncentered gradient
    gradnch = gradh .* exp(logzh);
    gradncJ = gradJ .* exp(logzJ);
    gradlogzh = gradh .* nch .* exp(logzh);
    gradlogzJ = gradJ .* ncJ .* exp(logzJ);

    % Add prior terms to form gradLogP, grad(+logPosterior)
    % Standard normals for non-centered parameters
    gradnch = gradnch - nch;
    gradncJ = gradncJ - ncJ;
    
    % Gradient due to the hyperprior
    gradhyper = zeros(size(hyper));
    
    % Horseshoe prior
    if strcmp(hyperprior, 'horseshoe')
        % Scale parameters are stored unconstrained
        scaleH = exp(hyper(1));
        scaleJ = exp(hyper(2));

        % Contributions from the local priors
        gradlogzh = gradlogzh + 1 - 2 * exp(2 * logzh) ./ (scaleH^2 + exp(2 * logzh));
        gradlogzJ = gradlogzJ + 1 - 2 * exp(2 * logzJ) ./ (scaleJ^2 + exp(2 * logzJ));
        gradhyper(1) = N - sum(2 * scaleH^2 ./ (scaleH^2 + exp(2 * logzh)));
        gradhyper(2) = N*(N-1)/2 - sum(2 * scaleJ^2 ./ (scaleJ^2 + exp(2 * squareform(logzJ))));

        % Contributions from the global prior
        gradhyper(1) = gradhyper(1) + 1 - 2 * scaleH^2 ./ (scaleH^2 + 1);
        gradhyper(2) = gradhyper(2) + 1 - 2 * scaleJ^2 ./ (scaleJ^2 + 1);
    end

    % Log-normal hyperprior
    if strcmp(hyperprior, 'lognormal')
        gradlogzh = gradlogzh - (logzh - hyper(1)) / exp(2 * hyper(3));
        gradlogzJ = gradlogzJ - (logzJ - hyper(2)) / exp(2 * hyper(4));
        gradhyper(1) = sum((logzh - hyper(1)) / exp(2 * hyper(3)));
        gradhyper(2) = sum((squareform(logzJ) - hyper(2)) / exp(2 * hyper(4)));
        gradhyper(3) = -N + sum((logzh - hyper(1)).^2 / exp(2 * hyper(3)));
        gradhyper(4) = -N*(N-1)/2 + sum((squareform(logzJ) - hyper(2)).^2 / exp(2 * hyper(4)));
        
        % Contributions from the global prior
        gradhyper(3) = gradhyper(3) + 1 - 2 * exp(2 * hyper(3)) ./ (exp(2 * hyper(3)) + 1);
        gradhyper(4) = gradhyper(4) + 1 - 2 * exp(2 * hyper(4)) ./ (exp(2 * hyper(4)) + 1);
    end

    % Pack negative gradient, grad(-logPosterior)
    gradncJ(logical(eye(size(gradncJ)))) = 0;
    gradlogzJ(logical(eye(size(gradlogzJ)))) = 0;
    gradtheta = pack_params(-gradhyper, -gradnch, -gradncJ, -gradlogzh, -gradlogzJ);
end

function callback(theta, ~, iter)
% Optionally plot current state during inference
    if ~strcmp(plot_mode, 'off') && mod(iter, 100) == 0
        [mu_hyp, mu_nch, mu_ncJ, mu_logzh, mu_logzJ] = ...
            unpack_params(theta(1:end/2), N, num_global);
        [logsig_hyp, logsig_nch, logsig_ncJ, logsig_logzh, logsig_logzJ] = ...
            unpack_params(theta(end/2 + [1:end/2]), N, num_global);
        
        exp(mu_hyp + 0.5 * transform_sigma(logsig_hyp).^2)
        
        h = mu_nch .* exp(mu_logzh + 0.5 * transform_sigma(logsig_logzh).^2);
        J = mu_ncJ .* exp(mu_logzJ + 0.5 * transform_sigma(logsig_logzJ).^2);
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
        stdJ = exp(mu_logzJ);
        stdJ = stdJ - diag(diag(stdJ));
        imagesc(stdJ)
        caxis(max(stdJ(:)) * [-1 1])
        colorbar
        axis square
        title('Coupling std. deviations')
        
        subplot(1,3,3)
        hold on
        errorbar(h, 2 * exp(logsig_nch), 'bx')
        plot([1 numel(h)], [0 0], 'k');
        hold off
        ylim([-1 1] * max([1 max(abs(h))]));
        xlim([0 numel(h) + 1]);
        title('Fields 95% bounds')
        

        if strcmp(plot_mode, 'full')
            % Plot current state of Markov chains
            figure(2)
            clf
            subplot(1,3,1)
            plot_mosaic(x, floor(sqrt(num_particles)), ...
                        floor(sqrt(N)),floor(sqrt(N)))

            subplot(1,3,2)
            plot_couplings(cov(x'))
            % Plot data statistics versus model statistics
            subplot(1,3,3)
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

function theta = pack_params(globals, nch, ncJ, logzh, logzJ)
    % Reshape from parameters to vector
    theta = [globals(:); nch(:); squareform(ncJ)'; logzh(:); ...
             squareform(logzJ)']; % Unpack dense J for faster computation
end

function [globals, nch, ncJ, logzh, logzJ] = unpack_params(theta, N, num_global)
    % Reshape from vector to parameters
    globals = reshape(theta(1:num_global), [num_global 1]);
    nch = reshape(theta(num_global + [1:N]), [N 1]);
    ncJ = squareform(theta(num_global + N + [1:N*(N-1)/2]));
    logzh = reshape(theta(num_global + N + N*(N-1)/2 + [1:N]), [N 1]);
    logzJ = squareform(theta(num_global + N + N*(N-1)/2 + N + [1:N*(N-1)/2]));
end
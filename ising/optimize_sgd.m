function x = optimize_sgd(gradfun, x_init, n_iter, params, callback)
%OPTIMIZE_SG optimizes a stochastic objective with given an 
%  ubiased estimator of the gradient, GRADFUN. It implementes SGD and Adam.
%

% Default algorithm is vanilla SGD
if ~exist('params','var')
    params.method = 'SGD';
    params.alpha = 0.0001;
end

% Optional callback routine for visualization & progress reporting
cback = 0;
if exist('callback','var')
    cback = 1;
end

% Initialization
x = x_init;

% Stochastic gradient descent
if strcmp(params.method, 'SGD')
    for i = 1:n_iter
        % Update
        g = gradfun(x, i);
        x = x - params.alpha * g;

        % Callback
        if cback
            callback(x, g, i);
        end
    end
end

% Adam
if strcmp(params.method, 'Adam')
    g_mu = zeros(size(x));
    g_sq = zeros(size(x));
    for i = 1:n_iter
        % Update
        g = gradfun(x, i);
        g_mu = params.beta1 * g_mu + (1 - params.beta1) * g; 
        g_sq = params.beta2 * g_sq + (1 - params.beta2) * g.^2;
        alpha_t = params.alpha * sqrt(1 - params.beta2^i) ...
                                   / (1 - params.beta1^i);
        x = x - alpha_t * g_mu ./ sqrt(g_sq + 1E-8);

        % Callback
        if cback
            callback(x, g, i);
        end
    end
end

% Adam, annealead linearly in the second half of training
if strcmp(params.method, 'AdamAnnealHalf')
    g_mu = zeros(size(x));
    g_sq = zeros(size(x));
    beta1_prod = 1;
    beta2_prod = 1;
    for i = 1:n_iter
        % Update
        if i < n_iter/2
            alpha = params.alpha;
            beta1 = params.beta1;
        else
            alpha = params.alpha * 2 * (n_iter - i) / n_iter;
            beta1 = params.beta1 * 2 * (n_iter - i) / n_iter;
        end
        beta2 = params.beta2;
        beta1_prod = beta1_prod * beta1;
        beta2_prod = beta2_prod * beta2;
        alpha_t = alpha * sqrt(1 - beta2_prod) / (1 - beta1_prod);
        g = gradfun(x, i);
        g_mu = beta1 * g_mu + (1 - beta1) * g; 
        g_sq = beta2 * g_sq + (1 - beta2) * g.^2;
        x = x - alpha_t * g_mu ./ sqrt(g_sq + 1E-8);

        % Callback
        if cback
            callback(x, g, i);
        end
    end
end

% Adam, annealead linearly in the second half of training
if strcmp(params.method, 'AdamAnneal')
    g_mu = zeros(size(x));
    g_sq = zeros(size(x));
    beta1_prod = 1;
    beta2_prod = 1;
    for i = 1:n_iter
        % Update
        alpha = params.alpha * (n_iter - i) / n_iter;
        beta1 = params.beta1 * (n_iter - i) / n_iter;
        beta2 = params.beta2;
        beta1_prod = beta1_prod * beta1;
        beta2_prod = beta2_prod * beta2;
        alpha_t = alpha * sqrt(1 - beta2_prod) / (1 - beta1_prod);
        g = gradfun(x, i);
        g_mu = beta1 * g_mu + (1 - beta1) * g; 
        g_sq = beta2 * g_sq + (1 - beta2) * g.^2;
        x = x - alpha_t * g_mu ./ sqrt(g_sq + 1E-8);

        % Callback
        if cback
            callback(x, g, i);
        end
    end
end

end


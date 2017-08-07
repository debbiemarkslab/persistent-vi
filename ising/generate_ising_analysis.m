%% Approximate inference uses Mark Schmidt's optimization code
addpath(genpath('external/L1General'))
addpath(genpath('external/minFunc'))

%% Persistent Gibbs Sampling C code
mex -lm CFLAGS='-O3 -fPIC -std=c99' sample_ising.c

%% Generate ferromagnet experiments
N_replicates = 1;
N_samples = [500, 1000, 2000];

for replicate = 1:N_replicates
    [X, h, J] = build_dataset('ferrocube', 1E4);
    for N_sample = N_samples
        X_train = X(1:N_sample, :);
        batch_inference_methods;
        file_fn = ['results/ferrocube_' num2str(replicate) ...
            '_' num2str(N_sample)];
        save(file_fn)
    end
end

%% Generate spin glass experiments
N_replicates = 5;
N_samples = [500, 1000, 2000];

for replicate = 1:N_replicates
    [X, h, J] = build_dataset('spinglass', 1E4);
    for N_sample = N_samples
        X_train = X(1:N_sample, :);
        batch_inference_methods;
        file_fn = ['results/spinglass_' num2str(replicate) ...
            '_' num2str(N_sample)];
        save(file_fn)
    end
end
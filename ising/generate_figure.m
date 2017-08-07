%% Ferromagnet experiments
clear all
figure(1)
clf
subplot(2,1,1)

N_samples = [500, 1000, 2000];
methods = {'Mean-field', 'PL, decimation', 'PL, L_1 (10xCV)', ...
                'MPF, L_1(10xCV)', 'PCD-3, L_1', 'PSVI-3, Half-Cauchy'};
error = zeros(numel(methods), numel(N_samples));
prefix = 'results/ferrocube_1_';

for ix_sample = 1:numel(N_samples)
    load([prefix num2str(N_samples(ix_sample))])
    
    % Fit naive mean field model
    J_NMF = -(cov(X_train)^(-1));
    J_NMF = J_NMF - diag(diag(J_NMF));

    % Use posterior mean estimator for Fadeout
    mu_J_fadeout = params_fadeout.mu_ncJ .* exp(params_fadeout.mu_logzJ + ...
        0.5 * params_fadeout.sig_logzJ.^2);

    % Order methods

    J_set = {J_NMF, params_pld.J, params_pl.J, ...
             params_mpf.J, params_pcd.J, mu_J_fadeout};

    % Compute error
    for ix_method = 1:numel(J_set)
        error(ix_method, ix_sample) = ...
            sqrt(mean(squareform(J_set{ix_method} - J).^2));
    end
    
end

% Plot the graph
barh(error', 'EdgeColor', 'None')
set(gca,'yticklabels',N_samples)
set(gca,'xscale','log')
set(gca,'ydir','reverse')
xlabel('RMS error, couplings {\bf J}')
ylabel('Sample size')
grid on
box on

% cmap = [102, 45, 145;...
%         39,  170, 225; ...
%         141,198, 63;...
%         255, 222, 23;...
%         251,176, 64; ...
%         236, 0, 140] / 255;

% cmap = [102, 45, 145;...
%         251,176, 64; ...
%         141,198, 63;...
%         255, 222, 23;...
%         39,  170, 225; ...
%         236, 0, 140] / 255;

cmap = [39,  170, 225; ...
        255, 222, 23;...
        251,176, 64; ...
        141,198, 63;...
        102, 45, 145;...
        236, 0, 140] / 255;

colormap(cmap)

title('Ferromagnet, 4x4x4 cube')
xlim([min(error(:))*exp(-0.5), max(error(:))*exp(0.5)])
legend(methods,'Location','southeast')

%
subplot(2,1,2)
clear all

N_replicates = 5;
N_samples = [500, 1000, 2000];
methods = {'Mean-field', 'PL, decimation', 'PL, L_1 (10xCV)', ...
                'MPF, L_1(10xCV)', 'PCD, L_1', 'PSVI, Horseshoe (nc)'};
error = zeros(N_replicates, numel(methods), numel(N_samples));
for ix_replicate = 1:N_replicates
    prefix = ['results/spinglass_' num2str(ix_replicate) '_'];
    % prefix = 'results/ferrocube_1_'

    for ix_sample = 1:numel(N_samples)
        load([prefix num2str(N_samples(ix_sample))])

        % Fit naive mean field model
        J_NMF = -(cov(X_train)^(-1));
        J_NMF = J_NMF - diag(diag(J_NMF));

        % Use posterior mean estimator for Fadeout
        mu_J_fadeout = params_fadeout.mu_ncJ .* exp(params_fadeout.mu_logzJ + ...
            0.5 * params_fadeout.sig_logzJ.^2);

        % Order methods

        J_set = {J_NMF, params_pld.J, params_pl.J, ...
                 params_mpf.J, params_pcd.J, mu_J_fadeout};

        % Compute error
        for ix_method = 1:numel(J_set)
            error(ix_replicate, ix_method, ix_sample) = ...
                sqrt(mean(squareform(J_set{ix_method} - J).^2));
        end

    end
end

% Average together
log_error_mean = squeeze(mean(log(error),1));
log_error_std = squeeze(std(log(error),1));
% Plot the graph
hold on
hb = barh(exp(log_error_mean'), 'EdgeColor', 'None');
xlim(exp([min(log_error_mean(:))-1.5, max(log_error_mean(:)) + 1.5]))
set(gca,'ytick',1:numel(N_samples))
set(gca,'yticklabels',N_samples)
set(gca,'xscale','log')
set(gca,'ydir','reverse')
xlabel('RMS error, couplings {\bf J}')
ylabel('Sample size')
grid on

cmap = [39,  170, 225; ...
        255, 222, 23;...
        251,176, 64; ...
        141,198, 63;...
        102, 45, 145;...
        236, 0, 140] / 255;

colormap(cmap)

title('Spin glass, ER topology (N=100, p=0.02)')
ix_counter = 1;
for ix_sample = 1:numel(N_samples)
    for ix_method = 1:numel(J_set)
        x = log_error_mean(ix_method, ix_sample);
        y = hb(ix_method).XData(ix_sample) + hb(ix_method).XOffset;
        width = 2 * log_error_std(ix_method, ix_sample);
        edges = exp(x + [-width width]);
        % Blend with white
        col = (cmap(ix_method,:) + 1.0) / 2.0;
        plot(edges, [y y], 'color', col, 'linewidth', 2);
        ix_counter = ix_counter + 1;
    end
end
hold off
box on
% %% Test case: Construct a random system with spin glass
% %
% L = 50;
% % q = 20;J = 0.75;
% p = 0.1;
% q = 20;
% % J = 0.8;
% J = 1.2;
% 
% h_alpha = 1;
% h_beta = 0.8;
% e_alpha = 1.2;
% e_beta = 1/0.1;
% load chain_50.mat
% 
% % Construct a parameter set with a spin-glass interactions
% 
% h_sigma = zeros(L,1);
% e_sigma = zeros(L);
% 
% % Inverse gamma distributed variances for h
% h_sigma = sqrt(1 ./ gamrnd(h_alpha, h_beta, L, 1));
% 
% %
% % Load the simulated protein topology
% %
% C = 1./D;
% C(C < prctile(C(:), (1-p)*100)) = 0;
% C(C > 0) = 1;
% e_sigma = C;
% 
% % Create the parameter set
% hi = zeros(L,q);
% eij = zeros(L,L,q,q);
% for i = 1:L
%     hi(i,:) = normrnd(0, h_sigma(i), q, 1);
% end
% 
% for i = 1:(L-1)
%     for j = (i+1):L
%         if e_sigma(i,j) > 0
%             eij(i, j, :, :) = normrnd(0,J,q,q);
%             eij(j, i, :, :) = squeeze(eij(i, j, :, :))';
%         end
%     end
% end
% 
% figure(1)
% subplot(2,1,1)
% bar(h_sigma)
% subplot(2,1,2)
% imagesc(e_sigma)
% colorbar
% axis equal
% axis tight
% 
% %%
% mex -lm CFLAGS='-O3 -fPIC -std=c99 -msse4.2' gibbs_potts.c
% 
% %%
% % Full Gibbs sampling of potts model with hi & eij
% % 1E4, 5E4
% N_full = 5E4;
% [sample_full, energies] = gibbs_potts(hi, eij, 1E4, N_full);
% 
% %%
% % Reduce autocorrelation to every kth sample
% N_full = size(sample_full,1);
% k = 50;
% sample = sample_full(k*[1:(N_full/k)],:);
% N = N_full / k;
% 
% %% Compare autocorrelation in samples before and after thinning
% figure(1)
% subplot(2,1,1)
% plot_autocorr_potts(sample_full)
% title('Autocorrelation from MCMC samples')
% axis tight
% grid on
% subplot(2,1,2)
% plot_autocorr_potts(sample)
% title('Autocorrelation in thinned dataset')
% axis tight
% grid on

%% Load dataset
load original_dataset

%% PVI calls
pvi = 'pvi/bin/pvi';

%% Cross-Validate Group L1 shrinkage
% k-fold cross validation
lg_range = [0.3 1.0 3.0 10.0 30.0 100.0];
lg_test = zeros(size(lg_range));
lg_train = zeros(size(lg_range));

tic
for ilg = 1:numel(lg_range)
    k = 5;
    n = 400;
    test_nlp = zeros(k,1);
    train_nlp = zeros(k,1);

    % Print a mock alignment 
    residues = ['-' 'A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' ...
                'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y']';
    % Divide data into k folds
    lbl = mod(randperm(n),k) + 1;
    
    for ik = 1:k
        fprintf('Lambda %f, fold %d\n', ilg, ik); 
        % Write fold ik
        fid = fopen(['results/cv' num2str(n, '%d') '.a2m'],'w');
        for i = find(lbl ~= ik)
            fprintf(fid, ['>' num2str(i) '\n']);
            fprintf(fid, [residues(sample(i,:))' '\n']);
        end
        fclose(fid);

        % Estimate the parameters
        system([pvi ' -o ali'  num2str(n, '%d') '_lg.eij -le 0.0 -lg ' num2str(lg_range(ilg)) ' -lh 0.1 -m 200 -t -1 -s 1.0 -a ' residues(1:q)' ' results/cv' num2str(n, '%d') '.a2m']);
        [hi_est_lg, eij_est_lg, ~, ~, ~] = read_eij(['ali'  num2str(n, '%d') '_lg.eij']);

        fprintf('Lambda %f, fold %d\n', ilg, ik);
        train_nlp(ik) = negative_log_pseudolikelihood(sample(lbl ~= ik,:), hi_est_lg, eij_est_lg) / sum(lbl ~= ik);
        test_nlp(ik) = negative_log_pseudolikelihood(sample(lbl == ik,:), hi_est_lg, eij_est_lg) / sum(lbl == ik);
    end
    lg_test(ilg) = mean(test_nlp);
    lg_train(ilg) = mean(train_nlp);
end
time_lg = toc;

%% Cross-Validate L2 shrinkage
% k-fold cross validation
l2_range = [0.3 1.0 3.0 10.0 30.0 100.0];
l2_test = zeros(size(l2_range));
l2_train = zeros(size(l2_range));

tic
for il2 = 1:numel(l2_range)
    k = 5;
    n = 400;
    test_nlp = zeros(k,1);
    train_nlp = zeros(k,1);

    % Print a mock alignment 
    residues = ['-' 'A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' ...
                'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y']';

    % Divide data into k folds
    lbl = mod(randperm(n),k) + 1;
    
    for ik = 1:k
        fprintf('Lambda %f, fold %d\n', il2, ik); 
        % Write fold ik
        fid = fopen(['results/cv' num2str(n, '%d') '.a2m'],'w');
        for i = find(lbl ~= ik)
            fprintf(fid, ['>' num2str(i) '\n']);
            fprintf(fid, [residues(sample(i,:))' '\n']);
        end
        fclose(fid)

        % Estimate the parameters
        system([pvi ' -o results/ali'  num2str(n, '%d') '_l2.eij -le ' num2str(l2_range(il2)) ' -lh 0.1 -m 200 -t -1 -s 1.0 -a ' residues(1:q)' ' results/cv' num2str(n, '%d') '.a2m']);
        [hi_est_l2, eij_est_l2, ~, ~, ~] = read_eij(['results/ali'  num2str(n, '%d') '_l2.eij']);

        fprintf('Lambda %f, fold %d\n', il2, ik);
        train_nlp(ik) = negative_log_pseudolikelihood(sample(lbl ~= ik,:), hi_est_l2, eij_est_l2) / sum(lbl ~= ik);
        test_nlp(ik) = negative_log_pseudolikelihood(sample(lbl == ik,:), hi_est_l2, eij_est_l2) / sum(lbl == ik);
    end
    l2_test(il2) = mean(test_nlp);
    l2_train(il2) = mean(train_nlp);
    [l2_range(il2) l2_test(il2) l2_train(il2)]
end
time_l2 = toc;

%% Plot Group L1 cross-validation results
figure(3)
clf
hold on
plot(lg_range, lg_test);
plot(lg_range, lg_train);
set(gca,'xscale','log');
xlim([min(lg_range) max(lg_range)]);
hold off
grid on
xlabel('Group L_1 regularization \lambda_G')
ylabel('Negative log pseudolikelihood')
legend({'5xCV', 'Training data'})

%% Assess L2 cross-validation results
%Elapsed time is 527.437036 seconds for lh0.1. best perf @ le10.0
figure(4)
clf
hold on
plot(l2_range, l2_test);
plot(l2_range, l2_train);
set(gca,'xscale','log');
xlim([min(l2_range) max(l2_range)]);
hold off
grid on
xlabel('L_2 regularization \lambda_J')
ylabel('Negative log pseudolikelihood')
legend({'5xCV', 'Training data'})

%% Determine optimal hyperparameters
[~, il2_min] = min(l2_test);
opt_l2 = l2_range(il2_min);
[~, ilg_min] = min(lg_test);
opt_lg = lg_range(ilg_min);

%% Write alignment to file
n = 400;
% 2-10 state Potts alphabet
residues = ['-' 'A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' ...
            'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y']';

fn_alignment = ['results/ali' num2str(n, '%d') '.a2m'];
fid = fopen(fn_alignment,'w');
for i = 1:n
    fprintf(fid, ['>' num2str(i) '\n']);
    fprintf(fid, [residues(sample(i,:))' '\n']);
end
fclose(fid);

%% Infer MAP estimate
% Estimate the parameters
system([pvi ' -o results/ali'  num2str(n, '%d') '_l2.eij -le ' num2str(opt_l2) ' -lh 0.1 -lg 0.0 -m 200 -t -1 -s 1.0 -a ' residues(1:q)' ' ' fn_alignment]);
[hi_est_l2, eij_est_l2, ~, ~, ~] = read_eij(['results/ali'  num2str(n, '%d') '_l2.eij']);

%% Infer Group-L1 regularized MAP estimate
% Estimate the parameters
system([pvi ' -o results/ali'  num2str(n, '%d') '_lg.eij -le 0.0 -lg ' num2str(opt_lg) ' -lh 0.1 -m 200 -t -1 -s 1.0 -a ' residues(1:q)' ' ' fn_alignment]);
[hi_est_lg, eij_est_lg, ~, ~, ~] = read_eij(['results/ali'  num2str(n, '%d') '_lg.eij']);

%% Infer Group-L1 regularized MAP estimate and visualize
% Estimate the parameters
system([pvi ' -o results/ali'  num2str(n, '%d') '_lg_manual.eij -le 0.0 -lg 30.0 -lh 0.1 -m 200 -t -1 -s 1.0 -a ' residues(1:q)' ' ' fn_alignment]);
[hi_est_lg_manual, eij_est_lg_manual, ~, ~, ~] = read_eij(['results/ali'  num2str(n, '%d') '_lg_manual.eij']);

%% Infer variationally approximated Hierarchical estimate and visualize
% Estimate the parameters
system([pvi ' -o results/ali'  num2str(n, '%d') '_vb.eij -v -vs 1 -gc 40 -gs 10 -m 5000 -t -1 -s 1.0 -a ' residues(1:q)' ' ' fn_alignment]);
params_vb = read_eij_bayes_full(['results/ali'  num2str(n, '%d') '_vb.eij']);

%%
% 313 seconds with best at LG = 10.0
% 289 seconds with best at L2 = 10.0

%% Performance
% CV L2 3.0 [31314.8879530831] 527.4
% CV LG 10.0 [23897.9584307954]  608.5
% LG 30.0 [24013.0745651086] N/A
% Variational Bayes [22946.3673858971] 597.2 ELBO 29619.4

%% Plot the coupling strengths for different methods
FN_true = plot_norms_eij(eij);
FN_l2 = plot_norms_eij(eij_est_l2);
FN_lg = plot_norms_eij(eij_est_lg);
FN_lg_manual = plot_norms_eij(eij_est_lg_manual);
FN_vb = plot_norms_eij_bayes(params_vb);

FN_set = {FN_true, FN_l2, FN_lg, FN_lg_manual, FN_vb};

% Determine color scale
cscale = 0;
for i = 1:numel(FN_set)
    cscale = max([cscale max(FN_set{i})]);
end

figure(1)
clf
% Plot various methods
for i = 1:numel(FN_set)
    subplot(1, numel(FN_set), i)
    imagesc(FN_set{i})
    axis square
    set(gca, 'TickLength', [0 0], 'XTickLabel', '', 'YTickLabel', '')
    caxis([0 cscale])
end
load ddblues
colormap(ddblues)

%% Compare the TP rates of the different methods
figure(5)
clf
cmap = [39,  170, 225; ...
        255, 222, 23;...
        251,176, 64; ...
        141,198, 63;...
        102, 45, 145;...
        236, 0, 140] / 255;
hold on

% Plot L2-regularized model
FN = plot_norms_eij(eij_est_l2);
plot_tp(C, FN, cmap(1,:))
% Plot Group Lasso-regularized model
FN = plot_norms_eij(eij_est_lg);
plot_tp(C, FN, cmap(4,:))
% Plot Group Lasso-regularized model, manual
FN = plot_norms_eij(eij_est_lg_manual);
plot_tp(C, FN, cmap(3,:))
% Plot hierarchical model
FN = plot_norms_eij_bayes(params_vb);
plot_tp(C, FN, cmap(6,:))

xlim([0 sum(C(:))/2])
hold off
xlabel('Top N scores')
ylabel('Fraction correct')
legend({['PL, L_2 (5xCV), \lambda_J = ' num2str(opt_l2)], ...
        ['PL, Group L_1 (5xCV), \lambda_G = ' num2str(opt_l2)], ...
        'PL, Group L_1, \lambda_G = 30.0', ...
        'PSVI-10, Half-Cauchy'}, 'Location', 'southwest')

box on
grid on

%% Compare performance on hold-out data
[hi_est_vb, eij_est_vb] = posterior_mean(params_vb);
%
NLP_l2lgvb = [negative_log_pseudolikelihood(sample(401:end,:), hi_est_l2, eij_est_l2)
              negative_log_pseudolikelihood(sample(401:end,:), hi_est_lg, eij_est_lg)
              negative_log_pseudolikelihood(sample(401:end,:), hi_est_vb, eij_est_vb)] / (1600);
          
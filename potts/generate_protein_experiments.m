%% Fit the SH3 data with Persistent SVI
system('pvi/bin/pvi -o results/PF00018_halfcauchy.vij -v -vs 10 -gc 40 -gs 10 -m 10000  protein_data/PF00018.a2m')

%% TODO: Enable Multivariate Laplace priors without recompilation
% pvi/bin/pvi -o results/PF00018_exponential.vij -v -vs 10 -gc 40 -gs 10 -m 10000  protein_data/PF00018.a2m

%% L2 regularized pseudolikelihood
system('pvi/bin/pvi -o results/PF00018_le9.6_lh0.01_m200.eij -le 9.6 -lh 0.01 -m 200 protein_data/PF00018.a2m')
[~, eij_est_l2, ~, ~, ~] = read_eij('results/PF00018_le9.6_lh0.01_m200.eij');
plot_norms_eij(eij_est_l2);

%% Group L1 regularized pseudolikelihood
system('pvi/bin/pvi -o results/PF00018_le0_lg30.0_lh0.01_m200.eij -lh 0.01 -lg 30 -le 0 -m 200 protein_data/PF00018.a2m')
[~, eij_est_lg, ~, ~, ~] = read_eij('results/PF00018_le0_lg30.0_lh0.01_m200.eij');
plot_norms_eij(eij_est_lg);


%% Load distance summaries
% for x in [min_dist[pair], mean_dist[pair], ca_dist[pair]]:
%    f.write('\t'.join([str(a) for a in [np.mean(x), np.min(x), np.median(x), np.max(x), np.std(x)]]) + '\t')
% I J COVERAGE [ mean min median max std ] 
% So medianMIN is the 6th column

X = dlmread('protein_data/PF00018_summary.txt');
I = floor(X(:,1));
J = floor(X(:,2));
medMIN = X(:,6);
D = zeros(size(FN_l2));
for ix = 1:numel(I)
    D(I(ix),J(ix)) = medMIN(ix);
    D(J(ix),I(ix)) = medMIN(ix);
end

% Inverse log colormap
figure(2)
imagesc(D)
axis square
caxis([0 max(D(:))])
set(gca, 'TickLength', [0 0])
% blumap = flipdim(cbrewer('seq', 'Blues', 512),1);
blumap = cbrewer('seq', 'Blues', 512);
blumap = blumap(floor(exp(linspace(0, log(512),100))), :);
colormap(flipdim(blumap,1))
colorbar

%%
% Load VB results
params_vb_exp = read_eij_bayes_full('results/PF00018_exponential.vij');
params_vb_hc = read_eij_bayes_full('results/PF00018_halfcauchy.vij');
FN_vb_exp = plot_norms_eij_bayes(params_vb_exp);
FN_vb_hc = plot_norms_eij_bayes(params_vb_hc);
% Load PL results
[~, eij_est_l2, ~, ~, ~] = read_eij('results/PF00018_le9.6_lh0.01_m200.eij');
[hi_est_lg, eij_est_lg, ~, ~, ~] = read_eij('results/PF00018_le0_lg30.0_lh0.01_m200.eij');
FN_l2 = plot_norms_eij(eij_est_l2);
FN_lg = plot_norms_eij(eij_est_lg);

FN_set = {FN_l2, FN_lg, FN_vb_exp, FN_vb_hc};
name_set = {'PL, L_2 \lambda_J = 9.6', 'PL, Group L_1 \lambda_G = 30.0', 'PSVI-10, Exponential', 'PSVI-10, Half-Cauchy'};

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

colormap(cbrewer('seq', 'Blues', 128));

load ddblues
colormap(ddblues)

set(gcf,'color','w')

figure(2)
clf
% TP rates
cmap = [39,  170, 225; ...
        255, 222, 23;...
        251,176, 64; ...
        141,198, 63;...
        102, 45, 145;...
        236, 0, 140] / 255;

cutoff = 10;
hold on
plot_tp(D < cutoff, FN_l2, cmap(1,:))
plot_tp(D < cutoff, FN_lg, cmap(4,:))
plot_tp(D < cutoff, FN_vb_exp, cmap(6,:))
plot_tp(D < cutoff, FN_vb_hc, cmap(5,:))
hold off
xlim([0 200])
xlabel('Top N interactions')
ylabel('Fraction < 10 Å')
box on
grid on
legend(name_set, 'Location', 'southwest');

%% Write PML for SH3 all scores with dynamic widths and colors
%

FN_set = {FN_l2, FN_lg, FN_vb_exp, FN_vb_hc};
suffix_set = {'pl_l2', 'pl_lg', 'vb_exp', 'vb_hc'};

% Determine color scale
cscale = 0;
for i = 1:numel(FN_set)
    cscale = max([cscale max(FN_set{i})]);
end

pdb = '1NYG'; % NMR solution structure
offset = 87; % Pfam is 88 - 135
chain = 'A';
num_bins = 8;
max_width = 15.0;


for FN_ix = 1:numel(FN_set)
    scores = FN_set{FN_ix}; 
    fn = ['results/' char(pdb) '_FN_' suffix_set{FN_ix} '.pml'];

    % Build a list of scores
    [II, JJ] = ndgrid(1:size(scores, 1), 1:size(scores, 1));
    ix = (II > JJ);
    ij = [II(ix) JJ(ix)];
    radius = scores(ix);

    % Highest bin contains upper edge
    bin_ix = floor(radius / cscale * num_bins);
    bin_ix(bin_ix == num_bins) = num_bins - 1;
    bin_ix = bin_ix + 1;
    bin_sizes = [0:(num_bins-1)] / (num_bins - 1) * max_width;

    % Build colormap
    load ddblues
    colmap = interp1(1:128, ddblues,...
                   linspace(1, 128, num_bins), 'linear');
    
    fid = fopen(fn,'w');

    custom = ['bg_color white\n'...
        'set ray_shadows, 0\n'...
        'set ambient, 0.5\n'...
        'set direct, 0\n'...
        'set spec_count, 0\n'... 
        'set cartoon_flat_sheets, 0\n'...
        'set cartoon_highlight_color = grey60\n'];

    pml = ['fetch ' pdb ', async=0\n' ...
        'hide all\n' ...
        'center chain ' chain '\n' ...
        'color tv_orange, chain ' chain '\n' ...
        'show cartoon, chain ' chain '\n' ...
        custom ...
        'set dash_gap, 0.0\n'];

    % Don't draw smallest bin (0 width)
    for i = 2:num_bins
        group_name = num2str(i);
        for subi = find(bin_ix == i)'
            resi = num2str(ij(subi,1) + offset);
            resj = num2str(ij(subi,2) + offset);
            pml = [pml ['dist dash_ec_group_' group_name ', chain ' chain ' and name ca and resid ' resi...
                ', chain ' chain ' and name ca and resid ' resj ', label=0\n']];
        end
        pml = [pml 'set_color col' group_name ', [ ' num2str(colmap(i,1),'%.5f') ', ' num2str(colmap(i,2),'%.5f') ', ' num2str(colmap(i,3),'%.5f') ' ]\n'];
        pml = [pml 'color col' group_name ', dash_ec_group_' group_name '\n'];
        pml = [pml 'set dash_width, ' num2str(bin_sizes(i),'%.5f') ', dash_ec_group_' group_name '\n'];
    end

    pml = [pml 'set_view (-0.206148207,    0.944744051,   -0.254877090, 0.562857091,   -0.098580010,   -0.820653915,-0.800433218,   -0.312637210,   -0.511434913,0.000000000,    0.000000000, -101.803634644,0.339273453,    1.821681023,   -4.362043381, 80.262756348,  123.344512939,  -20.000000000 )'];
    
    fprintf(fid, pml);
    fclose(fid);
end
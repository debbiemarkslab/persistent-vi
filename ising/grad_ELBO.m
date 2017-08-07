% SIGMA = EXP(U) Parameterization
function gradtheta = grad_ELBO(theta, gradNLP, num_samples)
% Compute a noisy estimate of the gradient of the negative ELBO under a
%  mean-field Gaussian variational approximation
     mu = theta(1:end/2);
     logsig = theta(end/2 + [1:end/2]);
     gradmu = zeros(size(mu));
     gradlogsig = zeros(size(mu));
     for i = 1:num_samples
         % Sample from the variational distribution
         Z = randn(size(mu));
         S = mu + exp(logsig) .* Z;

         % Negative log posterior for sample
         gradS = gradNLP(S);
         
         % Negative log posterior for sample
         gradmu = gradmu + gradS;
         gradlogsig = gradlogsig + gradS .* Z .* exp(logsig);
     end
     
     % Average over the samples and incorporate entropy term
     gradmu = gradmu / num_samples;
     gradlogsig = gradlogsig / num_samples - 1;

     gradtheta = [gradmu(:); gradlogsig(:)];
end

% SIGMA = LOG(1 + EXP(U)) Parameterization
% function gradtheta = gradELBO(theta, gradNLP, num_samples)
% % Compute a noisy estimate of the gradient of the negative ELBO under a
% %  mean-field Gaussian variational approximation
%      mu = theta(1:end/2);
%      logsig = theta(end/2 + [1:end/2]);
%      gradmu = zeros(size(mu));
%      gradlogsig = zeros(size(mu));
%      for i = 1:num_samples
%          % Sample from the variational distribution
%          Z = randn(size(mu));
%          S = mu + log(1 + exp(logsig)) .* Z;
% 
%          % Negative log posterior for sample
%          gradS = gradNLP(S);
%          
%          % Negative log posterior for sample
%          gradmu = gradmu + gradS;
%          gradlogsig = gradlogsig + gradS .* Z ./ (1 + exp(-logsig));
%      end
%      
%      % Average over the samples and incorporate entropy term
%      gradmu = gradmu / num_samples;
%      gradlogsig = gradlogsig / num_samples ...
%          - 1 ./ (log(1 + exp(logsig)) .* (1 + exp(-logsig)));
%      
%      gradtheta = [gradmu(:); gradlogsig(:)];
% end
clear all; clc; close all;

%% HMM INITIALIZATION

N = 100;                % number of observations

% Probability matrix
P = [0.8  0.1  0.1;
     0.2  0.5  0.3;
     0.3  0.1  0.6];

% States of the latent variables (mean)
mu = [1, 2, 3];

% Noise in the observation data (sigma)
sigma = [0.33, 0.33, 0.33];

%% Generate Markov Data

% Generate observations given original and hidden markov states
[x, z] = markovprocess(P, sigma, mu, N);

% Container to hold the estimate of z from x
est_x = zeros(size(x));

% Define thresholds to map noisy data to an estimated state variable
for l = 1:length(x)
    if x(l) >= 2.5
        est_x(l) = 3;
    elseif (x(l) >= 1.5) && (x(l) < 2.5)
        est_x(l) = 2;
    else
        est_x(l) = 1;
    end
end

% Relative error for estimated states
err_states = abs(est_x' - z);

% Latent states
Z = est_x';

%% 

% % Estimate P matrix from estimation of original state data (est_x)
% EP = zeros(size(P));        % transmission matrix
% 
% % Use counter to identify estimated state transitions
% % row => current state | column => future transition state (n+1)
% for i = 1:(length(Z) - 1)
%     EP(Z(i), Z(i+1)) = EP(Z(i), Z(i+1)) + 1;
% end
% 
% % Normalize each row by dividing by the total number of occurances for
% % state rows given by number of nonzero matrix elements (nnz)
% for k = 1:3
%    EP(k,:)  = EP(k,:)/sum(nnz(z==k));
% end

% Identify the probability of each latent state
N1=find(z==1); p_z1=length(N1)/length(N); % p(z=1)
N1=find(z==2); p_z2=length(N1)/length(N); % p(z=2)
N1=find(z==3); p_z3=length(N1)/length(N); % p(z=3)

% Estimate the transition probability from latent state z
n=zeros(size(P));
for i=1:N-1
    if (z(i)==1) && (z(i+1)==1)    
    n(1,1)=n(1,1)+1;
    elseif (z(i)==1) && (z(i+1)==2)    
    n(1,2)=n(1,2)+1;
    elseif (z(i)==1) && (z(i+1)==3)    
    n(1,3)=n(1,3)+1;
    elseif (z(i)==2) && (z(i+1)==1)    
    n(2,1)=n(2,1)+1;
    elseif (z(i)==2) && (z(i+1)==2)    
    n(2,2)=n(2,2)+1;
    elseif (z(i)==2) && (z(i+1)==3)    
    n(2,3)=n(2,3)+1;
    elseif (z(i)==3) && (z(i+1)==1)    
    n(3,1)=n(3,1)+1;
    elseif (z(i)==3) && (z(i+1)==2)    
    n(3,2)=n(3,2)+1;
    else    
    n(3,3)=n(3,3)+1;
    end 
end

% Normalize each row by the total number of occurances for each transition
n1=n(1,1)+n(1,2)+n(1,3); n(1,:)=n(1,:)/n1;
n2=n(2,1)+n(2,2)+n(2,3); n(2,:)=n(2,:)/n2;
n3=n(3,1)+n(3,2)+n(3,3); n(3,:)=n(3,:)/n3;

% Estimated transition probability matrix
EP=n;

disp("Estimated Transition Matrix: "); disp(EP);
disp("Original Transition Matrix: "); disp(P);

% Initialize model
model.E = EP;                % start probability vector
model.A = EP;                % transition matrix
model.s = [.33; .33; .33];   % emission matrix

% Compute alpha, beta, and gamma values
[gamma,alpha,beta,c] = hmmSmoother(model, Z);

% Find where transitions occured by looking at the absolute sum of
% probabilities
z_alpha = zeros(3,length(z));
z_beta = zeros(3,length(z));
z_gamma = zeros(3,length(z));
for j = 1:length(z)
    z_alpha(z(j),j) = 1;
    z_beta(z(j),j) = 1;
    z_gamma(z(j),j) = 1;
end

dif_alpha = z_alpha - alpha;
te_alpha = sum(abs(dif_alpha));

dif_beta = z_beta - beta;
te_beta = sum(abs(dif_beta));

dif_gamma = z_gamma - gamma;
te_gamma = sum(abs(dif_gamma));

%% FORWARD FILTERING

% Predict the states from the alpha values
z_alpha_gu = zeros(size(z));
for m = 1:length(alpha)
    [A,I] = max(alpha(:,m));
    z_alpha_gu(m) = I;
end

% Difference between the true latent hidden markov states and the predicted
% states using alpha values (forward filtering)
te_alpha_gu = sum(sum(abs(z_alpha_gu - z)));
fprintf('Forward prediction error: %i \n', te_alpha_gu);

% Represent the estimated states from the alpha values (forward filtering)
figure(1)
subplot(2,1,1), stem(z_alpha_gu);
title('State Prediction using Forward Filtering');
xlabel('Samples'), ylabel('States');
subplot(2,1,2), stem(z);
title('True Latent Markov States');
xlabel('Samples'), ylabel('States');

%% BACKWARD FILTERING

% Predict the states from the beta values (backward filtering)
z_beta_gu = zeros(size(z));
for m = 1:length(beta)
    [A,I] = max(beta(:,m));
    z_beta_gu(m) = I;
end

% Difference between the true latent hidden markov states and the predicted
% states using beta values (backward filtering)
te_beta_gu = sum(sum(abs(z_beta_gu - z)));
fprintf('Backward prediction error: %i \n', te_beta_gu);


% Represent the estimated states from the beta values (backward filtering)
figure(2)
subplot(2,1,1), stem(z_alpha_gu);
title('State Prediction using Backward Filtering');
xlabel('Samples'), ylabel('States');
subplot(2,1,2), stem(z);
title('True Latent Markov States');
xlabel('Samples'), ylabel('States');

%% FORWARD-BACKWARD SMOOTHING

% Predict the states from the gamma values
z_gamma_gu = zeros(size(z));
for m = 1:length(gamma)
    [A,I] = max(gamma(:,m));
    z_gamma_gu(m) = I;
end

% Difference between the true latent hidden markov states and the predicted
% states using gamma values (backward-forward smoothing)
te_gamma_gu = sum(sum(abs(z_gamma_gu - z)));
fprintf('Backward-Forward algorithm error: %i \n', te_gamma_gu);


% Represent the estimated states from the gamma values (backward-forward smoothing)
figure(3);
subplot(2,1,1), stem(z_gamma_gu);
title('State Prediction using Forward-Backward Smoothing');
xlabel('Samples'), ylabel('States');
subplot(2,1,2), stem(z);
title('True Latent Markov States');
xlabel('Samples'), ylabel('States');

%% BAUM-WELCH ALGORITHM

% Use the Baum-Welch algorithm to estimate gammas, giving a more accurate 
% estimate of the latent variables
[EMmodel, llh, gamma3] = hmmEm(Z, model);

% Latent variable prediction using the gamma values from the new model
z_EM = zeros(size(z));
for m = 1:length(gamma3)
    [A,I] = max(gamma3(:,m));
    z_EM(m) = I;
end

te_gamma_EM_gu = sum(sum(abs(z_EM - z)));
fprintf('Baum-Welch prediction error: %i \n', te_gamma_EM_gu);


% Represent the estimated states from the gamma values (Baum-Welch algorithm)
figure(4)
subplot(2,1,1), stem(z_EM);
title('State Prediction using the Baum-Welch Algorithm');
xlabel('Samples'), ylabel('States');
subplot(2,1,2), stem(z);
title('True Latent Markov States');
xlabel('Samples'), ylabel('States');

dif_gamma_guess_x = abs(z_EM - x');

sig_gu = zeros(size(sigma));
for p = 1:length(z_EM)
    if z_EM(p) == 1
       sig_gu(1,1) = sig_gu(1,1) + dif_gamma_guess_x(p);
    elseif z_EM(p) == 2
       sig_gu(1,2) = sig_gu(1,2) + dif_gamma_guess_x(p);       
    elseif z_EM(p) == 3
       sig_gu(1,3) = sig_gu(1,3) + dif_gamma_guess_x(p);
    end
end
    
for k = 1:3
   sig_gu(1,k)  = sig_gu(1,k)/sum(nnz(z_EM==k));
end

% Estimated P matrix
Est_P = EMmodel.A;

% Actual P matrix
P;

% Estimated sigmas
sig_gu;

% True sigmas
sigma;
    
%% VITERBI ALGORITHM

% Use the Baum-Welch model to create a latent variable vector and
% compare it to the true latent variable vector.
[z_new, llh] = hmmViterbi(EMmodel, Z);
%[z_new, llh] = hmmViterbi(EMmodel, z);

% Compare variable vectors
pred_close_or = Z - z;
pred_close_new = z_new - z;

%This is the comparison for the original observation using the thresholding
off_am_or = sum(sum(abs(pred_close_or)));
fprintf('Original prediction erro: %i \n', off_am_or);

%This is the comparison for the viterbi estimation
off_am_new = sum(sum(abs(pred_close_new)));
fprintf('Viterbi prediction error: %i \n', off_am_new);


% Represent the estimated states from the gamma values (Viterbi algorithm)
figure(5)
subplot(2,1,1), stem(z_new);
title('State Prediction using the Baum-Welch Algorithm');
xlabel('Samples'), ylabel('States');
subplot(2,1,2), stem(z);
title('True Latent Markov States');
xlabel('Samples'), ylabel('States');

%% Baum Welch estimated transmission P matrix and sigmas

% Estimated P matrix
disp('Baum Welch Estimated Transmission Matrix:');
disp(Est_P)

% True P matrix
disp('Original transmission matrix:');
disp(P);

% Estimated sigmas
disp('Baum Welch estimated sigmas:');
disp(sig_gu);

% True sigmas
disp('True Sigmas');
disp(sigma);


%% --------------------------- DEPRECATED --------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% FILTERING: Forward Algorithm
% % Calculate the alpha values to predict the transition probabilities
% % at each state given the current time.
% 
% % Compute the alpha values by filtering
% [alpha, ~] = hmmFilter(model, Z);
% 
% % For alpha we look at the absolute sum of the probabilities that we get
% % from the prediction. This is not very useful except it shows where the
% % transitions occured.
% z_alpha = zeros(3,length(z));
% for j = 1:length(Z)
%     z_alpha(Z(j),j) = 1;
% end
% 
% dif_alpha = z_alpha-alpha;
% te_alpha = sum(abs(dif_alpha));
% 
% % Predict the state from the alpha values
% z_alpha_gu = zeros(size(z));
% for m = 1:length(alpha)
%     [A,I] = max(alpha(:,m));
%     z_alpha_gu(m) = I;
% end
% 
% 
% %This is the difference between the true latent hidden markov states and
% %the guess we created from the alphas.
% te_alpha_gu = sum(sum(abs(z_alpha_gu - z)))
% 
% % Represent the states using the forwards algorithm
% figure(4);
% subplot(2,1,1), stem(z_alpha_gu);
% subplot(2,1,2), stem(z);

% %% Backward Algorithm
% [~, beta] = hmmFilter(model, Z);
% 
% z_beta = zeros(3,length(z));
% for j = 1:length(Z)
%     z_beta(Z(j),j) = 1;
% end
% 
% dif_beta = z_beta - beta;
% te_beta = sum(sum(abs(dif_beta)));
% 
% z_beta_gu = zeros(size(z));
% for m = 1:length(beta)
%     [A,I] = max(beta(:,m));
%     z_beta_gu(m) = I;
% end
% 
% %This is the difference between the true latent hidden markov states and
% %the guess we created from the betas.
% te_beta_gu = sum(sum(abs(z_beta_gu - z)))
% 
% figure(2);
% subplot(2,1,1), stem(z_beta_gu);
% subplot(2,1,2), stem(z);


% %% SMOOTHING: Forward-Backward Algorithm
% 
% [gamma2, alpha2, beta2, c2] = hmmSmoother(model, Z);
% 
% z_beta = zeros(3,length(z));
% for j = 1:length(Z)
%     z_beta(Z(j),j) = 1;
% end
% 
% dif_beta = z_beta - beta2;
% te_beta = sum(sum(abs(dif_beta)));
% 
% z_beta_gu = zeros(size(z));
% for m = 1:length(beta2)
%     [A,I] = max(beta2(:,m));
%     z_beta_gu(m) = I;
% end
% 
% %This is the difference between the true latent hidden markov states and
% %the guess we created from the betas.
% te_beta_gu = sum(sum(abs(z_beta_gu - z)))
% 
% figure(5);
% subplot(2,1,1), stem(z_beta_gu);
% subplot(2,1,2), stem(z);

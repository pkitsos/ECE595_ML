clear all; clc; close all;

%% Generate data for three 2D Gaussian distributions
mu1 = [1 2];
sigma1 = [3 1; 1 2];
n1 = 100;

mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
n2 = 100;

mu3 = [3 -3];
sigma3 = [1 .3; .3 1];
n3 = 200;

% X = [ mvnrnd(mu1, sigma1,n1);
%       mvnrnd(mu2, sigma2,n2);
%       mvnrnd(mu3, sigma3,n3)];

X1 = mvnrnd(mu1, sigma1,n1);
X2 = mvnrnd(mu2, sigma2,n2);
X3 = mvnrnd(mu3, sigma3,n3);
X = [X1; X2; X3];

%% Represent original data and associated pdf's

% Grid of coordinates for representation in the 2D space
x=linspace(-6,6,30);
x=repmat(x,length(x),1);
y=x';

% Vectorization of the coordinates
z=[x(:),y(:)];

%-------------------------------------------------------------------------%
% Compute pdf at each point of z
% Calculate the Gaussian response for every value in the grid.
z1 = mvgaussianPDF(z, mu1, sigma1);
z2 = mvgaussianPDF(z, mu2, sigma2);
z3 = mvgaussianPDF(z, mu3, sigma3);
Z = [z1;z2;z3];

%-------------------------------------------------------------------------%

% % Plots the data
% figure(1)
% set(gcf, 'Position', get(0,'Screensize'));
% % subplot(1,2,1); scatter(X(:,1),X(:,2),40,'ko', 'filled') 
% subplot(1,2,1); scatter(X1(:,1),X1(:,2),40,'ro', 'filled') 
% hold on
% scatter(X2(:,1),X2(:,2),40,'bo', 'filled') 
% scatter(X3(:,1),X3(:,2),40,'ko', 'filled') 
% contour(x,y,buffer(z1,sqrt(length(z1)),0))
% contour(x,y,buffer(z2,sqrt(length(z2)),0))
% contour(x,y,buffer(z3,sqrt(length(z3)),0))
% title('Original Data and PDFs of Gaussian Mixture Model');
% hold off
 

% Represents the PDF in a 3D plot
% subplot(1,2,2);
% surf(x,y,buffer(z1,sqrt(length(z1)),0),opts); hold on;
% subplot(1,2,2);surf(x,y,buffer(z2,sqrt(length(z2)),0),opts);
% subplot(1,2,2);surf(x,y,buffer(z3,sqrt(length(z3)),0),opts);
% title('Original PDF of Gaussian Mixture Model');
% axis tight; view(-50,30); camlight left;


opts.FaceColor = 'interp';
opts.EdgeColor = 'none';
opts.FaceLighting = 'Phong';
%% Initialize the mixture model parameters

% Set 'm' to the number of data points
m = size(X, 1);

k = 3;  % Number of clusters
n = 2;  % Vector lengths

% Initialize means by randomly selecting k data points
idx = randperm(m);
mu = X(idx(1:k), :);

sigma = [];

% Overall covariance of dataset used as the initial variance for each cluster
for (j = 1 : k)
    sigma{j} = cov(X);
end

% Assign equal prior probabilities to each cluster
pi = ones(1, k) * (1 / k);


%% Begin Expectation Maximization algorithm

% Matrix holds the responsibility probability that each data point 
% belongs to a cluster (Rows => data points |  Columns => clusters)
R_ik = zeros(m, k);
keepLn=[];
Lo=10;
% Loop until convergence

for (iter = 1:1000)
    
    fprintf('EM Iteration %d\n', iter);

    %% Compute Expectation   
    % Calculate the probability for each data point for each cluster
    
    % Matrix to hold the pdf value for each data point in every cluster
    % (Rows => data points |  Columns => clusters)
    pdf = zeros(m, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian PDF for all data points in cluster 'j'
        pdf(:, j) = mvgaussianPDF(X, mu(j, :), sigma{j});
    end
    
    % Multiply each pdf value by the prior probability for cluster
    % (pdf  [m  x  k] | phi  [1  x  k] | pdf_w  [m  x  k])
    pdf_r = bsxfun(@times, pdf, pi);
    
    % Compute the responsibilities by dividing the weighted probabilities
    % by the sum of weighted probabilities for each cluster
    %   sum(pdf_w, 2) -- sum over the 2D clusters
    R_ik = bsxfun(@rdivide, pdf_r, sum(pdf_r, 2));
    
    
    %% Maximize the computed Expectation
    % Calculate the probability for each data point for each distribution

    % Store the previous means
    prevMu = mu;    
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior for cluster 'j'
        pi(j) = mean(R_ik(:, j), 1);
        
        % Calculate new mean for cluster 'j' by taking the weighted
        % average of all data points
        mu(j, :) = weightedAvg(R_ik(:, j), X);

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example 
        
        sigma_k = zeros(n, n);
        
        % Subtract the cluster mean from all data points
        Xm = bsxfun(@minus, X, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix
        for (i = 1 : m)
            sigma_k = sigma_k + (R_ik(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        
        % Divide by the sum of responsibilities
        sigma{j} = sigma_k ./ sum(R_ik(:, j));
    end
    
    k=3;
    W=pi;
    M=mu';
    V=sigma;
    Ln = Likelihood(X,k,W,M,V);
    keepLn=[keepLn Ln];

    % Check for convergence when computed mean no longer changes
%     if (mu == prevMu)
%         break
%     end
    
    % Check for convergence when computed mean no longer changes
    if (abs(Lo-Ln)<0.001)
        break
    end
     Lo = Ln;
    %% Represent mixture model
    
    % Calculate the Gaussian response for every value in the grid
    z1 = mvgaussianPDF(z, mu(1, :), sigma{1});
    z2 = mvgaussianPDF(z, mu(2, :), sigma{2});
    z3 = mvgaussianPDF(z, mu(3, :), sigma{3});
    Z = [z1;z2;z3];

    % Plot the data and estimated pdfs for each cluster
    figure(1)
    subplot(1,2,1); 
    box on;
    scatter(X1(:,1),X1(:,2),40,'ro', 'filled'); hold on;
    scatter(X2(:,1),X2(:,2),40,'bo', 'filled'); 
    scatter(X3(:,1),X3(:,2),40,'co', 'filled'); 
    % Represent the true cluster means
    plot(mu1(1), mu1(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
    plot(mu2(1), mu2(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
    plot(mu3(1), mu3(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
    contour(x,y,buffer(z1,sqrt(length(z1)),0), 'LineWidth', 1.5);
    contour(x,y,buffer(z2,sqrt(length(z2)),0), 'LineWidth', 1.5); 
	contour(x,y,buffer(z3,sqrt(length(z3)),0), 'LineWidth', 1.5); hold off;
    % Format plot
    legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'True Mean', 'Location', 'best');
    title('Original Data and Estimated PDFs of Gaussian Mixture Model', 'Fontsize', 12);
    xlabel('X', 'FontSize', 12); ylabel('Y', 'FontSize', 12);


    % Plot 3D PDF
    subplot(1,2,2); 
    surf(x,y,buffer(z1,sqrt(length(z1)),0),opts); hold on;
    surf(x,y,buffer(z2,sqrt(length(z2)),0),opts);
    surf(x,y,buffer(z3,sqrt(length(z3)),0),opts); hold off;
    % Format plot
    title('Estimated PDFs of Gaussian Mixture Model', 'FontSize', 12);
    xlabel('X', 'FontSize', 12); ylabel('Y', 'FontSize', 12); zlabel('Probability', 'FontSize', 12);
    axis tight; view(-50,30); camlight left;

    % Maximize figure to screensize
    set(gcf, 'Position', get(0,'Screensize'));

    %fprintf(' Mu1: (%d,%d) Mu2: (%d,%d) Mu3: (%d,%d)\n', mu1, mu2, mu3);

    % End of Expectation Maximization    
end

% Ouputs:
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%   L - log likelihood of estimates
function L = Likelihood(X,k,W,M,V)
% Compute L based on K. V. Mardia, "Multivariate Analysis", Academic Press, 1979, PP. 96-97
% to enchance computational speed
[n,d] = size(X);
U = mean(X)';
S = cov(X);
L = 0;
for i=1:k,
    iV = V{i};
    L = L + W(i)*(-0.5*n*log(det(2*pi*V{i})) ...
        -0.5*(n-1)*(trace(iV*S)+(U-M(:,i))'*iV*(U-M(:,i))));
end
end
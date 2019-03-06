function [ pdf ] = mvgaussianPDF(X, mu, Sigma)
% DESCRIPTION:
%       Computes the PDF of a 2D multivariate Gaussian distribution
%
% INPUT:
%       X: Matrix of data points (one per row)
%       mu: Row vector for the mean
%       Sigma: Covariance matrix
%
% OUTPUT:
%       pdf: PDF of multivariate Gaussian

% Get vector length
n = size(X, 2);

% Subtract the mean from every data point
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));

end
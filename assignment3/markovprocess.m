% DESCRIPTION:
%       Implements a Hidden Markov Model (HMM) with a number 'k' of hidden
%       states 'z_n' and a univariate random number 'x' whose mean and
%       variance depend on the value of the hidden state.
% INPUT:
%       P: transition probability matrix
%       sigma: standard deviation of distribution
%       mu: mean of distribution
%       N: number of observations
% OUTPUT:
%       x: univariate random variable depending on value of hidden state
%       z: true latent hidden states

function [x,z]=markovprocess(P,sigma,mu,N)
p=cumsum(P,2);
zact=ceil(rand*length(mu));
z=[];
for i=1:N
    a=rand;
    zact=[min(find(p(zact,:)>=a))];
    z=[z zact];
end
x=randn(size(z)).*sigma(z)+mu(z);
x=x';
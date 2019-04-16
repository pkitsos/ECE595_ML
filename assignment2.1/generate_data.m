%% Assignment 2.1       
%% ------------    Array Data Simulation   ------------------ %%
function [X,B]=generate_data(N,n_elements,phi,A,sigma)
%Inputs:
% N: Number of samples
% n_elements: Number of antennas (corresponds to dimension D)
% phi: Vector of L sources
% A: Vector of amplitudes for the L sources.
% sigma: Noise standard deviation.
%
%Outputs
% X: Matrix of snapshots.
% B: Matrix of signals (not relevant in this assignment).
X=0;
B=[];
for i=1:length(A)
%Produce a set of N complex baseband data fo direction i
[x,b]=data(N,n_elements,phi(i));
%Multiply it by its amplitude
X=X+x*A(i);
B=[B b];
end
%Add noise
X=X+sigma*(randn(size(X))+1j*randn(size(X)))/2;
function [x,b]=data(N,n_elements,phi)
%Simulate data in a N element array
x=zeros(n_elements,N); %Snapshots (row vectors)
b=sign(randn(N,1)) + 1j*sign(randn(N,1)); %Data
for i=1:N
x(:,i)=exp(1j*phi*(0:n_elements-1)')*b(i);
end

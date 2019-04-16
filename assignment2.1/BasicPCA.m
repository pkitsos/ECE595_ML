%% Assignment 2.1
%% ------------------- Basic PCA --------------------- %%
clear
close
clc
N = 100;                            %number of samples
M = 10;                             %number of array elements
phi = [pi/2,pi,pi*3/2];             %angles of arrival for L sources

[X,B] = generate_data(N,M,phi,1,3);

% Autocorrelation matrix calculation and its inverse
R = X*X'./N;
Rinv = inv(R);
% Eigenvector calculation
[V,L] = eig(R);               %V => eigenvectors, lam => eigenvalues

% ek = zeros(length(phi),M);      %steering vector
% omega = pi*sin(phi);            %spatial frequency
% S_mvdr = zeros(length(phi),1);  %
% S_music = zeros(length(phi),1);
% w_k = zeros(10,length(phi));
% for k = 1:length(omega)
%    for j = 1:M
%        ek(j,k) = exp((j-1)*omega(k)*1i);
%    end
%    S_mvdr(k) = 1/(ek(:,k)'*Rinv*ek(:,k));
%    S_music(k) = 1/(ek(:,k)'*(V*V')*ek(:,k));
%    w_k(:,k) = Rinv*ek(:,k)/(ek(:,k)'*Rinv*ek(:,k));
%    w_k(:,k)'*ek(:,k)
% end

%% MUSIC algorithm for Noise eigenvectors
V_noise = V(:,1:M-length(phi));         %Noise subspace of the eigenspace
V_signal = V(:,M-length(phi):end);      %Signal subspave of eigenvectors

theta = (-90:0.5:90)*pi/180;
P_music_n = zeros(1,length(theta));
P_music_s = zeros(1,length(theta));
for ii=1:length(theta)
    ek = zeros(1,M);                %Steering Vevtor
    for k=0:M-1
        %exp(-j2pi*d/lambda *sin(theta))
        ek(1+k) = exp(-1i*pi*k*sin(theta(ii)));
    end
    % For noise eigenvectors
    S_music = ek*(V_noise*V_noise')*ek';
    P_music_n(ii) = abs(1/S_music);
    % For Signal eigenvectors
    S_music = ek*(eye(length(V_signal))-V_signal*V_signal')*ek';
    P_music_s(ii) =abs(1/S_music);
end
% Spatial spectrum function
P_music_n=10*log10(P_music_n/max(P_music_n)); 
P_music_s=10*log10(P_music_s/max(P_music_s));

%% plots
%theta = theta*180/pi;
figure(1)
plot(theta,P_music_n)
xlabel('Normalized Frequency (x\pi rad/sample)')
ylabel('Power (dB)')
title('Pseudospectrum Estimate via MUSIC Noise EigenVectors')
grid on

figure(2)
plot(theta,P_music_s)
xlabel('Normalized Frequency (x\pi rad/sample)')
ylabel('Power (dB)')
title('Pseudospectrum Estimate via MUSIC Signal EigenVectors')
grid on

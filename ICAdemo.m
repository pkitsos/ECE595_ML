clear all

%data genration
x=(-14:14)/29; 
z1=kron(ones(1,18),x); 
z1=z1(1:500);
x=(-9:9)/25; 
z2=kron(ones(1,28),x.^3)/0.1;
z2=z2(1:500); 
z3=sin(2*pi*32*(0:499)/500)/2;
z4=randn(1,500); 
W=[1 1 0.5 0.2;0.2 1 1 0.5;0.2 1 1.4 0.4; 1 0.3 0.5 0.5];
X=W'*[z1;z2;5*z3;0.01*z4];


%Plot before FastICA
figure(1)
 hold on
 subplot(4,1,1); plot(squeeze(z1));
 subplot(4,1,2); plot(squeeze(z2));
 subplot(4,1,3); plot(squeeze(z3));
 subplot(4,1,4); plot(squeeze(z4));

%FastICA
V = W^-1;   %Recognition Matrix

%update V
for i=1:4
    V(:,i) = (X * (tanh((V(:,i))' * X)') - ((1 - tanh((V(:,i))' * X).^2) * X')')/4;

%force orthogonality
%ignored

%normalize V
%expectation already implements normalization

end

%latent variable updated
z1 = V(1,:) * X;
z2 = V(2,:) * X;
z3 = V(3,:) * X;
z4 = V(4,:) * X;


%Plot after FastICA is applied
figure(2)
 hold on
 subplot(4,1,1); plot(squeeze(z1));
 subplot(4,1,2); plot(squeeze(z2));
 subplot(4,1,3); plot(squeeze(z3));
 subplot(4,1,4); plot(squeeze(z4));
 

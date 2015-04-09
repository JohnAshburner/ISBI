
sd = round(rand(1)*100);
sd=37;
randn('seed',sd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulated data
%----------------
Sig = randn(5,2);Sig=Sig'*Sig/5; % Covariance
mu  = randn(2,2)*2; % means;
N1 = 10;
N2 = 20;
N  = N1+N2;
X  = zeros(2,N1+N2);
X(:,    1:N1) = repmat(mu(:,1),1,N1) + sqrtm(Sig)*randn(2,N1);
X(:,(N1+1):(N1+N2)) = repmat(mu(:,2),1,N2) + sqrtm(Sig)*randn(2,N2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mn = min(X');
mx = max(X');
d  = mx-mn;
x1 = (1:512)/512*d(1)+mn(1);
x2 = (1:512)/512*d(2)+mn(2);

r1 = 64;
r2 = 449;
s = [1 0 r1; 0 1 r1; 1 0 r2; 0 1 r2]\[mn'; mx'];

x1 = (1:512)*s(3) + s(1);
x2 = (1:512)*s(3) + s(2);
mn = [min(x1),min(x2)];
mx = [max(x1),max(x2)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show "Ground Truth"
clf
iSig = inv(Sig);
[X1,X2] = ndgrid(x1-mu(1,1),x2-mu(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Sig));
[X1,X2] = ndgrid(x1-mu(1,2),x2-mu(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Sig));
scal = 64/max(p1(:)+p2(:));


subplot(2,2,1);
imagesc(x1,x2,p1'./(p1'+p2')); set(gca,'Clim',[0 1]);
hold on
contour(x1,x2,scal*p2',0:8:64,'r');
contour(x1,x2,scal*p1',0:8:64,'r');

a = Sig\(mu(:,2)-mu(:,1));
b = -a'*(mu(:,1)+mu(:,2))/2;
o = null(a')*mean(diff(x1));
c = mean(X,2);

z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'w','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);
title('Ground truth')
xlabel('Feature 1')
ylabel('Feature 2')
drawnow
print -depsc simple_ground_truth.eps
!epstopdf simple_ground_truth.eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show Fisher's Linear Discriminant
p = [[ones(1,N1); zeros(1,N1)] [zeros(1,N2); ones(1,N2)]];
C = zeros(2);
for k=1:2,
    m(:,k) = sum(X.*repmat(p(k,:),2,1),2)/sum(p(k,:));
    tmp    = X - repmat(m(:,k),1,N);
    C      = C + (repmat(p(k,:),2,1).*tmp)*tmp';
end
C    = C/(N1+N2-2);
iSig = inv(C);
[X1,X2] = ndgrid(x1-m(1,1),x2-m(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(C));
[X1,X2] = ndgrid(x1-m(1,2),x2-m(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(C));

subplot(2,2,1);
image(x1,x2,scal*p1');
hold on
contour(x1,x2,scal*p1',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=0) = p(x|y=0) p(y=0)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,2);
image(x1,x2,scal*p2');
hold on
contour(x1,x2,scal*p2',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=1) = p(x|y=1) p(y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,3);
image(x1,x2,scal*(p1'+p2'));
hold on
contour(x1,x2,scal*(p1'+p2'),0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x) = p(x,y=0) + p(x,y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,4);
image(x1,x2,64*p1'./(p1'+p2'));
hold on
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(y=0|x) = p(x,y=0)/p(x)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

colormap(1-gray)
drawnow
print -depsc simple_fld.eps
!epstopdf simple_fld.eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show Quadratic Discriminant
p = [[ones(1,N1); zeros(1,N1)] [zeros(1,N2); ones(1,N2)]];
Cq = zeros(2,2,2);
for k=1:2,
    m(:,k) = sum(X.*repmat(p(k,:),2,1),2)/sum(p(k,:));
    tmp    = X - repmat(m(:,k),1,N);
    Cq(:,:,k) = (repmat(p(k,:),2,1).*tmp)*tmp';
end
Cq(:,:,1) = Cq(:,:,1)/(N1-2);
Cq(:,:,2) = Cq(:,:,2)/(N2-2);

iSig = zeros(2,2,2);
for k=1:2, iSig(:,:,k) = inv(Cq(:,:,k)); end

[X1,X2] = ndgrid(x1-m(1,1),x2-m(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1,1).*X1 + X1.*iSig(1,2,1).*X2 + X2.*iSig(2,1,1).*X1 + X2.*iSig(2,2,1).*X2))./sqrt((2*pi)^2*det(Cq(:,:,1)));
[X1,X2] = ndgrid(x1-m(1,2),x2-m(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1,2).*X1 + X1.*iSig(1,2,2).*X2 + X2.*iSig(2,1,2).*X1 + X2.*iSig(2,2,2).*X2))./sqrt((2*pi)^2*det(Cq(:,:,2)));

subplot(2,2,1);
image(x1,x2,scal*p1');
hold on
contour(x1,x2,scal*p1',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=0) = p(x|y=0) p(y=0)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,2);
image(x1,x2,scal*p2');
hold on
contour(x1,x2,scal*p2',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=1) = p(x|y=1) p(y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,3);
image(x1,x2,scal*(p1'+p2'));
hold on
contour(x1,x2,scal*(p1'+p2'),0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x) = p(x,y=0) + p(x,y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,4);
image(x1,x2,64*p1'./(p1'+p2'));
hold on
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(y=0|x) = p(x,y=0)/p(x)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

colormap(1-gray)
drawnow
print -depsc simple_qda.eps
!epstopdf simple_qda.eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show Naive Bayes
Cd = diag(diag(C));
iSig = inv(Cd);
[X1,X2] = ndgrid(x1-m(1,1),x2-m(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Cd));
[X1,X2] = ndgrid(x1-m(1,2),x2-m(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Cd));

subplot(2,2,1);
image(x1,x2,scal*p1');
hold on
contour(x1,x2,scal*p1',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=0) = p(x|y=0) p(y=0)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,2);
image(x1,x2,scal*p2');
hold on
contour(x1,x2,scal*p2',0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x,y=1) = p(x|y=1) p(y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,3);
image(x1,x2,scal*(p1'+p2'));
hold on
contour(x1,x2,scal*(p1'+p2'),0:8:64,'r');
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(x) = p(x,y=0) + p(x,y=1)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

subplot(2,2,4);
image(x1,x2,64*p1'./(p1'+p2'));
hold on
pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('p(y=0|x) = p(x,y=0)/p(x)')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy

colormap(1-gray)
drawnow
print -depsc simple_naive_bayes.eps
!epstopdf simple_naive_bayes.eps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf
subplot(2,2,1);
[X1,X2] = ndgrid(x1,x2);
XX=X'*X;
w=svc(XX,p(1,:)'*2-1,1,1000);
P = ones(size(X1))*w(end);
for i=1:size(X,2),
    P = P + w(i)*X(1,i)'*X1 + w(i)*X(2,i)'*X2;
end
P = P>0;
imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
hold on

a = X*w(1:(end-1));
b = w(end);
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'w','LineWidth',2);
o   = null(a');
plot([-100; 100],[( 1 - b + 100*a(1))/a(2); ( 1 - b - 100*a(1))/a(2)],'k',...
     [-100; 100],[(-1 - b + 100*a(1))/a(2); (-1 - b - 100*a(1))/a(2)],'w','LineWidth',1);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

ind = find(abs(w)>1e-3); ind=ind(ind<=size(p,2));
plot(X(1,ind),X(2,ind),'r*');

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);
title('SVC')
xlabel('Feature 1')
ylabel('Feature 2')

drawnow
print -depsc svc.eps
!epstopdf svc.eps











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure
colormap(1-gray)

iSig = inv(C);
[X1,X2] = ndgrid(x1-m(1,1),x2-m(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(C));
[X1,X2] = ndgrid(x1-m(1,2),x2-m(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(C));

subplot(2,2,2);
imagesc(x1,x2,p1'./(p1'+p2')); set(gca,'Clim',[0 1]);
hold on
contour(x1,x2,scal*p2',0:8:64,'k');
contour(x1,x2,scal*p1',0:8:64,'w');

a =   C\(m(:,2)-m(:,1));
b = -a'*(m(:,1)+m(:,2))/2;
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'w','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
title('LDA')
xlabel('Feature 1')
ylabel('Feature 2')
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);



iSig = inv(Sig);
[X1,X2] = ndgrid(x1-mu(1,1),x2-mu(2,1));
p1 = (N1/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Sig));
[X1,X2] = ndgrid(x1-mu(1,2),x2-mu(2,2));
p2 = (N2/(N1+N2))*exp(-0.5*(X1.*iSig(1,1).*X1 + X1.*iSig(1,2).*X2 + X2.*iSig(2,1).*X1 + X2.*iSig(2,2).*X2))./sqrt((2*pi)^2*det(Sig));
subplot(2,2,1);
imagesc(x1,x2,p1'./(p1'+p2')); set(gca,'Clim',[0 1]);
hold on
contour(x1,x2,scal*p2',0:8:64,'k');
contour(x1,x2,scal*p1',0:8:64,'w');

a = Sig\(mu(:,2)-mu(:,1));
b = -a'*(mu(:,1)+mu(:,2))/2;
o = null(a')*mean(diff(x1));
c = mean(X,2);

z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'w','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
 
hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);
title('Ground truth')
xlabel('Feature 1')
ylabel('Feature 2')



subplot(2,2,3);
[X1,X2] = ndgrid(x1,x2);
XX=X'*X;
w=svc(XX,p(1,:)'*2-1,1,1000);
P = ones(size(X1))*w(end);
for i=1:size(X,2),
    P = P + w(i)*X(1,i)'*X1 + w(i)*X(2,i)'*X2;
end
P = P>0;
imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
hold on

a = X*w(1:(end-1));
b = w(end);
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'w','LineWidth',2);
o   = null(a');
plot([-100; 100],[( 1 - b + 100*a(1))/a(2); ( 1 - b - 100*a(1))/a(2)],'k',...
     [-100; 100],[(-1 - b + 100*a(1))/a(2); (-1 - b - 100*a(1))/a(2)],'w','LineWidth',1);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

ind = find(abs(w)>1e-3); ind=ind(ind<=size(p,2));
plot(X(1,ind),X(2,ind),'r*');

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);
title('SVC')
xlabel('Feature 1')
ylabel('Feature 2')


subplot(2,2,4)
[X1,X2] = ndgrid(x1,x2);
grp = [1 1 2];
[w,al,ll]=logistic_ridge_regression([X',ones(size(X,2),1)],p(1,:)',grp);
P       = 1./(1+exp(-(X1*w(1)+X2*w(2)+w(3))));
imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
hold on
%contour(x1,x2,P',0.5,'r');
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'w',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'k','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);

title('Simple Logistic Regression')
xlabel('Feature 1')
ylabel('Feature 2')

if false
subplot(2,3,6);
[X1,X2] = ndgrid(x1,x2);
%mu = mean(X,2);
%[P]   = GP_pics(X-repmat(mu,1,size(X,2)),p(1,:)',X1-mu(1),X2-mu(2));
[P]   = GP_pics(X,p(1,:)',X1,X2);
%imagesc(x1,x2,P'); axis image xy
%hold on
imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
hold on
contour(x1,x2,P',0:0.05:1,'k');

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);
hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);
title('Gaussian Process')
xlabel('Feature 1')
ylabel('Feature 2')

end

drawnow
%print -deps linear_discrimination.eps



% Bayesian methods
clf

subplot(2,2,1)
[X1,X2] = ndgrid(x1,x2);
grp = [1 1 2];
hparam = zeros(max(grp),2)+1e-4;

[w,H,al,ll,Pb]=logistic_ridge_regression([X',ones(size(X,2),1)],p(1,:)',grp,hparam,[X1(:),X2(:), [], ones(numel(X1),1)]);
Pb = reshape(Pb,size(X1));
P  = 1./(1+exp(-(X1*w(1)+X2*w(2)+w(3))));
%imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
contour(x1,x2,P',0:0.05:1,'r');
hold on
%contour(x1,x2,P',0.5,'r');
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'k','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);

title('Simple Logistic Regression')
xlabel('Feature 1')
ylabel('Feature 2')




subplot(2,2,2)
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
%plot([c(1)-a(1)*100,c(1)+a(1)*100],[c(2)-a(2)*100,c(2)+a(2)*100],'k','LineWidth',2);
%hold on

for i=1:50,
    w1 = w + sqrtm(H)*randn(3,1);
    a  = w1(1:2);
    b  = w1(end);
    z  = -(a'*c + b)/(a'*a);
%   plot([c(1)-a(1)*100,c(1)+a(1)*100],[c(2)-a(2)*100,c(2)+a(2)*100],'k','LineWidth',1);
    plot([-100; 100],[(-b + 100*a(1))/a(2); (-b - 100*a(1))/a(2)],'b','LineWidth',0.5);
    hold on
end

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);


hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);

title('Hyperplane Uncertainty')
xlabel('Feature 1')
ylabel('Feature 2')


subplot(2,2,3)
imagesc(x1,x2,Pb');  set(gca,'Clim',[0 1]);
hold on
%contour(x1,x2,P',0.5,'r');
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'w',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'k','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);


title('Bayesian Logistic Regression')
xlabel('Feature 1')
ylabel('Feature 2')

subplot(2,2,4)
%imagesc(x1,x2,Pb');  set(gca,'Clim',[0 1]);
contour(x1,x2,Pb',0:0.05:1,'r');
hold on
%contour(x1,x2,P',0.5,'r');
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);
z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'k','LineWidth',2);

pl1 = plot(X(1,1:N1),X(2,1:N1),'ko'); set(pl1,'MarkerFaceColor',[1 1 1]);
pl2 = plot(X(1,(N1+1):(N1+N2)),X(2,(N1+1):(N1+N2)),'wo'); set(pl2,'MarkerFaceColor',[0 0 0]);

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);

title('Bayesian Logistic Regression')
xlabel('Feature 1')
ylabel('Feature 2')

drawnow
print -depsc logistic_regr.eps
!epstopdf logistic_regr.eps



clf
subplot(2,2,1);
[w,al,ll]=logistic_ridge_regression([X',ones(size(X,2),1)],p(1,:)',grp);
P       = 1./(1+exp(-(X1*w(1)+X2*w(2)+w(3))));
%imagesc(x1,x2,P');  set(gca,'Clim',[0 1]);
hold on
probs = [0.001,0.01,0.1,0.5,0.9,0.99,0.999];
contour(x1,x2,P',probs,'k');
hold on
a = w(1:2);
b = w(end);
o = null(a')*mean(diff(x1));
c = mean(X,2);

z = -(a'*c + b)/(a'*a);
plot([c(1)+a(1)*z,c(1)+a(1)*100],[c(2)+a(2)*z,c(2)+a(2)*100],'k',...
     [c(1)+a(1)*z,c(1)-a(1)*100],[c(2)+a(2)*z,c(2)-a(2)*100],'k','LineWidth',2);
plot(c(1),c(2),'+','MarkerSize',20,'MarkerFaceColor',[1 1 1]);
for i=1:numel(probs)
    z  = -(a'*c + b -log(1/(1-probs(i))-1))/(a'*a);
    a1 = a*z;
   %a1 = a/(a'*a)*(-log(1/(1-probs(i))-1) );
    plot(c(1)+a1(1),c(2)+a1(2),'o','MarkerSize',6,'MarkerFaceColor',[0 0 0]);
    text(c(1)+a1(1)+0.2,c(2)+a1(2)+0.2,num2str(probs(i)));
end

hold off
axis image xy
axis([mn(1),mx(1),mn(2),mx(2)]);

title('Caricatures [p(y=0|x)]')
xlabel('Feature 1')
ylabel('Feature 2')
drawnow
%print -deps caricatures.eps



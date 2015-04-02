function [w,S,alpha,ll,P]=logistic_ridge_regression(X,t,grp,hhparam,X0) 
% John Ashburner
% $Id$

if nargin<3 || isempty(grp),     grp     = ones(size(X,2),1); end
if nargin<4 || isempty(hhparam), hhparam = zeros(max(grp),2)+1e-4; end

[M,N]        = size(X);
%t            = logical(t(:));
alpha        = ones(max(grp),1)*100;
[w,alpha,ll,H] = lreg(X,t,alpha,grp,hhparam);
A            = zeros(N,1);

for g=1:max(grp)
    A(grp==g)  = alpha(g);
end
alpha = A;
%[w,alpha,ll,H] = lreg_ard(X,t,alpha);
S     = inv(H);

if nargin>3 && nargout>=5,
    msk = w~=0;
    wm  = w(msk);
    P   = zeros(size(X0,1),1);
    for i=1:numel(P),
        x0   = X0(i,:)';
        x0   = x0(msk);
        mu   = wm'*x0;
        sig2 = x0'*(H(msk,msk)\x0);
        kap  = (1+pi*sig2/8)^(-1/2);
        P(i) = lsig(kap*mu);
     end

    if false
        % GP mechanism (Laplace approx)
        P1  = zeros(size(X0,1),1);
        C   = X*diag(1./A)*X' + eye(M)*1e-4;
        msk = w~=0;
        wm  = w(msk);
        X1  = X(:,msk);
        sig = lsig(X*w);
        W   = diag(sig.*(1-sig)); % pg. 316
        iWC = inv(inv(W) + C);    % Eq. 6.88

        for i=1:numel(P1),
            x0   = X0(i,msk);
            k    = X1*diag(1./A)*x0';
            c    = x0*diag(1./A)*x0' + 1e-4;
            mu   = k'*(t-sig);
            vr   = c - k'*iWC*k;
            kap  = (1+pi*vr/8)^(-1/2);
            P1(i) = lsig(kap*mu);
         end
     end

end

return;
%__________________________________________________________________________

%__________________________________________________________________________
function [w,alpha,ll,H] = lreg(X,t,alpha,grp,hhp)
tin   = realmin;
[N,M] = size(X);
w     = zeros(size(X,2),1);
A     = zeros(M,1);
for g=1:max(grp)
    A(grp==g)  = max(alpha(g),tin);
end
nz    = true(size(X,2),1);
nz1   = true(size(alpha));
gama  = hhp(:,1);
gamb  = hhp(:,2);

alpha_old = alpha;
th=1;
for subit=1:512,
    alpha_old(nz1) = alpha(nz1);

   % E-step
    [w(nz),H,y] = map_soln(X(:,nz),t,A(nz),w(nz));
   %[w,H,y]     = map_soln(X,t,A,w);
%plot(t,X*w,'+'); drawnow

    % M-step
    A  = zeros(M,1);
    %ds = diag(inv(H));
    ds = zeros(M,1);ds(nz) = diag(inv(H));
    for g=1:max(grp)
        msk      = grp==g;
        alpha(g) = (sum(1-ds(msk)*alpha(g))+2*gama(g))/(sum(w(msk).^2)+2*gamb(g));
    end

    th = min(alpha)*1e8/N;
    nz1= alpha<th;
    A  = zeros(M,1);
    ds = zeros(M,1);
    ds(nz) = diag(inv(H));
    for g=1:max(grp)
        if ~nz1(g)
            alpha(g)   = th;
            w(grp==g)  = 0;
            nz(grp==g) = false;
        else
            nz(grp==g) = true;
        end
        A(grp==g)  = max(alpha(g),tin);
    end

    % Convergence
    if max(abs(log((alpha(nz1)+eps)./(alpha_old(nz1)+eps)))) < 1e-3,
        break
    end
   %fprintf('%d\t%d\t%g\t%g\n', subit, sum(nz), max(abs(log(alpha(nz1)+eps))),max(abs(log(alpha(nz1)+eps) - log(alpha_old(nz1)+eps))));
end

% Objective function for GP classification
%[w,H,y]    = map_soln(X,t,A,w);
[w(nz),H,y] = map_soln(X(:,nz),t,A(nz),w(nz));
w(~nz)      = 0;
C   = X(:,nz)*diag(1./A(nz))*X(:,nz)' + eye(N)*1e-8;
a   = X(:,nz)*w(nz);
%a  = mode_a(C,t,a);
sw  = sqrt(y.*(1-y));
ll  = -0.5*a'*(C\a) -0.5*logdet(eye(N)+(sw*sw').*C) + t'*a - sum(log(1+exp(a)));
for g=1:max(grp)
   %if nz1(g)
%        ll  = ll  + 0.5*sum(grp==g)*(gama(g)*log(alpha(g)+eps) - gamb(g)*alpha(g));
   %end
end
return;
%__________________________________________________________________________

%__________________________________________________________________________
function [w,H,y] = map_soln(X,t,alpha,w)
% MAP solution for w
% Also returns the formal covariance matrix of the fit on the assumption
% of normally distributed errors (inverse of Hessian), for Laplace
% approximation.

teeny = 1e-20;
[M,N] = size(X);
y     = lsig(X*w);
%err   = -sum(log(max(y(t),teeny)))-sum(log(max(1-y(~t),teeny))) + 0.5*(alpha'*(w.^2));
err   = -sum(t.*log(max(y,teeny)))-sum((1-t).*log(max(1-y,teeny))) + 0.5*(alpha'*(w.^2));

g     = X'*(y-t) + alpha.*w;
beta  = y.*(1-y);
H     = X'*diag(beta)*X + diag(alpha);

I   = eye(N);
lam = 1e-5;
lam = max(diag(H))*1e-10;
for it=1:1000,
    w_old   = w;
    err_old = err;
    w     = w - (H+I*lam)\g;
    y     = lsig(X*w);
   %err   = -sum(log(max(y(t),teeny)))-sum(log(max(1-y(~t),teeny))) + 0.5*(alpha'*(w.^2));
    err   = -sum(t.*log(max(y,teeny)))-sum((1-t).*log(max(1-y,teeny))) + 0.5*(alpha'*(w.^2));
   %fprintf('\t\t%d\t%g\t%g\t%g\t%d\n', it,err_old,err,lam,err>err_old);
    if err>err_old*(1+eps*N),
        w   = w_old;
        err = err_old;
        lam = lam*10;
    else
        lam  = max(lam/10,max(diag(H))*1e-12);
        g    = X'*(y-t) + alpha.*w;
        beta = y.*(1-y);
        H    = X'*diag(beta)*X + diag(alpha);
        if it>2 & norm(g)<N*1e-8,
            break;
        end
    end
end
return;
%__________________________________________________________________________

%__________________________________________________________________________
function [ld,C] = logdet(A)
A  = (A+A')/2;
C  = chol(A);
d  = max(diag(C),realmin);
ld = 2*sum(log(d));
%__________________________________________________________________________

%__________________________________________________________________________
function a = mode_a(C,t,a)
% Find mode of a, for Laplace approximation
% Note that it has been expressed as a minimisation of
% the negative log-likelihood, rather than a maximisation
% of the log-likelihood
N = numel(t);
if nargin<3, a = zeros(N,1); end;
for i=1:64,
    sig = lsig(a);
    g   = -(t - sig - C\a);  % -ve 1st deriv (Eq. 6.81)
    W   = diag(sig.*(1-sig));
    H   = (W + inv(C));      % -ve 2nd deriv (Eq. 6.82)
    a   = a - H\g;           % Newton-Raphson iteration
    if sum(g.^2)/sum(a.^2)<eps, break; end
end
%__________________________________________________________________________

%__________________________________________________________________________
function sig = lsig(a)
% Logistic sigmoid (Eq. 4.59)
sig = 1./(1+10000*eps+exp(-a));
%__________________________________________________________________________


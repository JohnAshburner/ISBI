
for ii=1:3,
switch ii
case 1,
rad   = sqrt(linspace(8^2,63^2,36));
case 2
rad   = linspace(8,63,36);
case 3
rad   = (logspace(log10(8),log10(63),36));
end
X     = zeros(128,128,36);
[x,y] = ndgrid(-64:63,-64:63);
r = sqrt(x.^2+y.^2);
for i=1:numel(rad),
    X(:,:,i) = ((r<rad(i))+(r<rad(i)+0.25)+(r<rad(i)-0.25))/3 - ((r<rad(i)/2)+(r<rad(i)/2+0.25)+(r<rad(i)/2-0.25))/3;
   %X(:,:,i) = spm_conv(X(:,:,i),4);
end

subplot(3,2,ii*2-1); montage(reshape(X,[128 128 1 36]))

X1=reshape(X,[128^2,36]);
[U,S,V]=svd(X1,0);
U1 = U*S;
imx=max(U1(:));
imn=min(U1(:));
U1 = (U1-imn)/(imx-imn);
subplot(3,2,ii*2); montage(reshape(U1,[128 128 1 36]))
end



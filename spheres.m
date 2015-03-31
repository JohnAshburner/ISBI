
r=randn(2,10)*0.8;
r0=r-repmat(mean(r,2),1,size(r,2));
aviobj = avifile('example.avi');

colormap(gray)
for oit = 1:2,
for t=1:40,
r=r0*t/40;
[x,y,z]=sphere(400);
s0=surf(x,y,z);
set(s0,'FaceAlpha',0.8,'SpecularStrength',0.5,'DiffuseStrength',1,'AmbientStrength',0.05);
shading interp
hold on

for i=1:size(r,2),
    l=zeros(3);
    l(3,1)=-r(1,i);
    l(1,3)= r(1,i);
    l(3,2)=-r(2,i);
    l(2,3)= r(2,i);
    t = zeros(10,3);
    for j=1:50,
        a=expm(l*(j-1)/50);
        t(j,:) = (a*[0 0 1.04]')';
    end
    p=plot3(t(:,1),t(:,2),t(:,3),'r');
    set(p,'LineWidth',2,'Color',[0.5 0 0]);
    xyz = a*[0 0 1.04]';
    [x1,y1,z1] = sphere(30);
    x1 = x1*0.04 + xyz(1);
    y1 = y1*0.04 + xyz(2);
    z1 = z1*0.04 + xyz(3);
    
    s=surf(x1,y1,z1); 
    set(s,'FaceColor',[1 0 0],'EdgeAlpha',[0],'FaceLighting','flat');

    if oit>1
    s=patch([-2 2 2 -2]',[-2 -2 2 2]',1.04*[1 1 1 1]',[1 1 1 1]');
    set(s,'FaceAlpha',0.1,'SpecularStrength',0.5,'DiffuseStrength',1,'AmbientStrength',0.05);
    set(s,'FaceColor',[1 1 1],'EdgeAlpha',[0],'FaceLighting','flat');

    p=plot3([0 r(1,i)]', [0 r(2,i)]', [1.04 1.04]', 'r'); hold on
    set(p,'LineWidth',2,'Color',[0 0 0.5]);
    [x1,y1,z1] = sphere(30);
    x1 = x1*0.04 + r(1,i);
    y1 = y1*0.04 + r(2,i);
    z1 = z1*0.04 + 1.04;
    s=surf(x1,y1,z1);
    set(s,'FaceColor',[0 0 1],'EdgeAlpha',[0],'FaceLighting','flat');
    end
end
axis image off
axis([-2 2 -2 2 -1.1 1.1])
hold off

set(gcf,'Color',[0 0 0]);
c=camlight;
set(c,'Position',[-15 -10 0],'Color',[0 0 0.5]);
c=camlight;
set(c,'Position',[-5 -5 15],'Color',[1 1 1]);
drawnow
F = getframe(gcf);
aviobj = addframe(aviobj,F);
end
end
aviobj = close(aviobj);

if false
clf
for i=1:size(r,2),
    l=zeros(3);
    l(3,1)= r(1,i);
    l(1,3)=-r(1,i);
    l(3,2)= r(2,i);
    l(2,3)=-r(2,i);
    t = zeros(10,3);
    p=plot3([0 r(1,i)]', [0 r(2,i)]', [0 0]', 'r'); hold on
    set(p,'LineWidth',2,'Color',[0.5 0 0]);
    [x1,y1,z1] = sphere(30);
    x1 = x1*0.04 + r(1,i);
    y1 = y1*0.04 + r(2,i);
    z1 = z1*0.04;
    s=surf(x1,y1,z1);
    set(s,'FaceColor',[1 0 0],'EdgeAlpha',[0],'FaceLighting','flat');
    hold on
end

hold off
axis image off
end


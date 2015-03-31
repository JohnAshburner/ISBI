clf
r=randn(2,6)*0.7;
r0=r-repmat(mean(r,2),1,size(r,2));

colormap(gray)
r=r0;
[x,y,z]=sphere(400);
s0=surf(x,y,z);
set(s0,'FaceAlpha',0.8,'SpecularStrength',0.5,'DiffuseStrength',1,'AmbientStrength',0.05);
shading interp
hold on

for i=1:size(r,2),
    li     = zeros(3);
    li(3,1)=-r(1,i);
    li(1,3)= r(1,i);
    li(3,2)=-r(2,i);
    li(2,3)= r(2,i);
    a_i    = expm(li);
    xyz = a_i*[0 0 1.04]';
    [x1,y1,z1] = sphere(30);
    x1 = x1*0.04 + xyz(1);
    y1 = y1*0.04 + xyz(2);
    z1 = z1*0.04 + xyz(3);
    s=surf(x1,y1,z1);    
    set(s,'FaceColor',[1 0 0],'EdgeAlpha',[0],'FaceLighting','flat');

    for j=1:size(r,2),
        lj     = zeros(3);
        lj(3,1)=-r(1,j);
        lj(1,3)= r(1,j);
        lj(3,2)=-r(2,j);
        lj(2,3)= r(2,j);
        a_j    = expm(lj);

        t = [];
        for s=0:0.01:1,
            a_s = logm(a_i\a_j);
            a   = a_i*expm(s*a_s);
            t = [t; (a*[0 0 1.04]')'];
        end
        p=plot3(t(:,1),t(:,2),t(:,3),'r');
        set(p,'LineWidth',2,'Color',[0.5 0 0]);
    end
end
    

%    if oit>1
%    s=patch([-2 2 2 -2]',[-2 -2 2 2]',1.04*[1 1 1 1]',[1 1 1 1]');
%    set(s,'FaceAlpha',0.1,'SpecularStrength',0.5,'DiffuseStrength',1,'AmbientStrength',0.05);
%    set(s,'FaceColor',[1 1 1],'EdgeAlpha',[0],'FaceLighting','flat');

%    p=plot3([0 r(1,i)]', [0 r(2,i)]', [1.04 1.04]', 'r'); hold on
%    set(p,'LineWidth',2,'Color',[0 0 0.5]);
%    [x1,y1,z1] = sphere(30);
%    x1 = x1*0.04 + r(1,i);
%    y1 = y1*0.04 + r(2,i);
%    z1 = z1*0.04 + 1.04;
%    s=surf(x1,y1,z1);
%    set(s,'FaceColor',[0 0 1],'EdgeAlpha',[0],'FaceLighting','flat');
%    end
%end
axis image off tight
%axis([-2 2 -2 2 -1.1 1.1])
hold off

%set(gcf,'Color',[0 0 0]);
c=camlight;
set(c,'Position',[-15 -10 0],'Color',[0 0 0.5]);
c=camlight;
set(c,'Position',[-5 -5 15],'Color',[1 1 1]);
drawnow



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


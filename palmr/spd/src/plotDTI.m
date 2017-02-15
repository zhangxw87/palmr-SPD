function [] = plotDTI(D)
% plotDTI plots the diffusion tensor imaging
%
%   plotDTI(D)
%   D is an array of size [3 3 nx ny]
%
% 
% Written by Xiaowei Zhang 
% 2015/05/20
% updated on 2017/02/14

sz=size(D);
if length(sz)==2
    nx=1;ny=1;
elseif length(sz)==3
    nx=sz(3);ny=1;
elseif length(sz)==4
    nx=sz(3);ny=sz(4);
end

for i = 1:nx
    for j = 1:ny
        [v,d] = eig(D(:,:,i,j));
        d = diag(d);
        [X,Y,Z] = ellipsoid(0,0,0,d(1),d(2),d(3));
        sz = size(X);
        for x = 1:sz(1)
            for y = 1:sz(2)
                A = [X(x,y) Y(x,y) Z(x,y)]';
                A = v*A;
                X(x,y) = A(1);
                Y(x,y) = A(2);
                Z(x,y) = A(3);
            end
        end
        subplot(ny,nx,(j-1)*nx+i);
        h(i,j) = surf(X,Y,Z);

        axis equal
        view([0 90]);
        set(gca,'GridLineStyle','none')
        set(gca,'XTick',[])
        set(gca,'YTick',[])
        set(gca,'ZTick',[])
        shading interp
        colormap(jet); % colormap(autumn)
        lighting phong
        light('Position',[0 0 1],'Style','infinite','Color',[ 1.000 0.584 0.000]);
        axis off
    end
end
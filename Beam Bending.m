clear all;
default = {'1'};
L = inputdlg('input the length of the beam L=','L=',1,default);
L = str2double(L);
EI =inputdlg('input the product of Young''s modulus and momoent of inertia of the beam EI=','EI=',1,default);
EI = str2double(EI);
default = {'-1'};
f = inputdlg('input the uniform stress f0=','f0=',1,default);
f = str2double(f);
default = {'0'};
fdelta = inputdlg('input the delta stress fdelta=','fdelta=',1,default);
fdelta = str2num(fdelta{1});
default = {'0.5'};
xx = inputdlg('input the effect position of delta stress x=','x=',1,default);
xx = str2num(xx{1});
[left,~] = listdlg('ListString',{'fixed','supported','free'},'Name','node','PromptString','what is the status of the left node?','SelectionMode','Single');
[right,~] = listdlg('ListString',{'fixed','supported','free'},'Name','node','PromptString','what is the status of the right node?','SelectionMode','Single');

n = 100; %number of elements
h = L/n;

%fd'' and fs''
fd2back = [-6,0,6]/h^2; %(0,1)
fd2forw = [6,0,-6]/h^2; %(-1,0)
fs2back = [-4,-1,2]/h^2; %(0,1)
fs2forw = [-2,1,4]/h^2; %(-1,0)

W=[h/6,h*4/6,h/6]';

%Elementry Stifness Matrix
func = [fd2back;fs2back;fd2forw;fs2forw];
E=zeros(4);
for i = 1:4
    for j = 1:4
        E(i,j) = func(i,:).*func(j,:)*W;
    end
end

%construct the global matrix
K = zeros(2*n+2,2*n+2); 
for i = 1:n 
    aa = 2*i-1; 
    %K(aa:aa+3,aa:aa+3) = E; 
    %因为这个是积分 所以是相加 （分区见积分 然后自然就是求和
    K(aa:aa+3,aa:aa+3) =K(aa:aa+3,aa:aa+3)+ E;
end

syms x
X=x/h;
phi0s= X* (abs(X)-1)^2;
X2=(x-h)/h;
phi1d=(abs(X2)-1)^2*(2*abs(X2)+1);

%Apply BC
if right == 1
% Built in fixed in Beam
% u =0 u'=0
%drop phid 0 phid N phis 0 phis N
    K(end,:)=[];
    K(:,end)=[];
    K(end,:)=[];
    K(:,end)=[];
end
if right == 2
% simple supported in Beam
% u =0
    K(end-1,:)=[];
    K(:,end-1)=[];
end
if left == 1
    K(1,:)=[];
    K(:,1)=[];
    K(1,:)=[];
    K(:,1)=[];
end
if left == 2 
    K(1,:)=[];
    K(:,1)=[];
end

 % assume F is a constant function

intphid=int(phi1d*f,[0,2*h]);
intphis=int(phi0s*f,[0,h]);
intphid=double(intphid);
intphis=double(intphis);

for i =1:2:2*(n-1) %from 1 2 3 4 ... N-1
    F(i)=intphid;
    F(i+1)=0;
end

fd=@(X)(abs(X)-1)^2*(2*abs(X)+1);
fs=@(X)X* (abs(X)-1)^2;

for t=1:length(fdelta)
    ii=floor(xx(t)*L/h); %in the i th interival the index is i+1 
    %from global to local
    X=(xx(t)/L-ii*h/L)/h;
    F(ii*2-1)= F(ii*2-1)+fd(X)*fdelta(t);
    F(ii*2)= F(ii*2)+fs(X)*fdelta(t);
    % 2 basis function will be influenced
    ii=ii+1;
    X=(xx(t)/L-ii*h/L)/h;
    F(ii*2-1)= F(ii*2-1)+fd(X)*fdelta(t);
    F(ii*2)= F(ii*2)+fs(X)*fdelta(t);
end

%BC
if right == 2
    F = [F,-intphis];
end
if right == 3
    F = [F,-intphis,-intphis];
end
if left == 2
    F = [intphis,F];
end
if left == 3
    F = [intphis,intphis,F];
end

U=K\F';

%Construct the origin function
fd=@(X)(abs(X)-1)^2*(2*abs(X)+1);
fd0=fd(0); %should be 1

%BC
if right == 2
    U = U(1:end-1);
end
if right == 3
    U = U(1:end-2);
end
if left == 2
    U = U(2:end);
end
if left == 3
    U = U(3:end);
end

u=[];
for i =1:2:2*n-2 %from 1 2 3 4 ... N-1
    u = [u,U(i)*fd0];   
end

%BC
if right ~= 3
    if left ~= 3
        u = [0,u,0];
    else
        u = [2*u(1)-u(2),u,0];
    end
else
    u = [0,u,2*u(end)-u(end-1)];
end
u = u*EI;

plot(0:h:L,u,'r')

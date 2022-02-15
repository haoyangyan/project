h = 11; %分支层数
load('data.mat')
x = cell(task,1);
opt = zeros(task,1);
% 松弛掉原问题的整数约束，求解LP 
for i = 1:task
    scoresi = length(data{i});
    f = [data{i}{:,3}]';
    V = [data{i}{:,2}];
    b = ones(length([data{i}{1,2}]),1);
    O = [data{i}{:,1}];
    beq = ones(length([data{i}{1,1}]),1);
    ub = ones(scoresi,1);
    lb = zeros(scoresi,1);
    xi = linprog(f,V,b,O,beq,lb,ub);
    if isempty(xi) == 1
        x{i} = NaN;
        opt(i) = NaN;
    end
    % 将近似0,1的x更新为0,1
    for j = 1:scoresi
        if abs(xi(j))<10^-5
            xi(j)=0;
        end
        if abs(xi(j)-1)<10^-5
            xi(j)=1;
        end
    end
    % 判定不等于0,1的x的位置
    undef = [];
    for j = 1:scoresi
        if xi(j) ~= 1
            if xi(j) ~= 0
                undef = [undef,j];
            end
        end
    end
    noun = length(undef);
    % 构建高度为h的二叉树
    issol = zeros(1,2^h-1); %是否有解树，0表示无整数解，1表示有整数解，2表示无解，3表示解超过最优值上界
    def = cell(1,2^h-1); %已确定订单编号树(仅在无整数解节点赋值)
    xdef = cell(1,2^h-1); %已确定订单取0还是1树(在所有子节点赋值)
    opti = inf * ones(1,2^h-1); %最优值树(仅在有整数解节点赋值)
    xxi = cell(1,2^h-1); %解树(在有解节点赋值)
    if noun == 0 %存在整数解
        issol(1) = 1;
        opti(1) = f'*xi;
        xxi{1} = xi;
    else %不存在整数解
        def{1} = undef(1); 
    end
    best = inf;
    for k = 2:2^h-1
        father = floor(k/2);
        if issol(father) == 0 %仅对父节点无整数解情况下往下分支
            if mod(k,2) == 0 %左儿子，对应让未确定订单取0
                xdef{k} = [xdef{father},0];
                ft = f;
                Vt = V;
                bt = b;
                Ot = O;
                beqt = beq;
                ubt = ub;
                lbt = lb;
                ft(def{father}) = [];
                Vt(:,def{father}) = [];
                Ot(:,def{father}) = [];
                del = def{father}.*xdef{k};
                del(del==0)=[];
                bt = bt-sum(V(:,del),2);
                beqt = beqt-sum(O(:,del),2);
                ubt(def{father}) = [];
                lbt(def{father}) = [];
                xit = linprog(ft,Vt,bt,Ot,beqt,lbt,ubt);
                if isempty(xit) == 1 %本支无解
                    issol(k) = 2;
                    continue
                end
                % 将近似0,1的x更新为0,1
                scoresik = length(xit);
                for j = 1:scoresik
                    if abs(xit(j))<10^-5
                        xit(j)=0;
                    end
                    if abs(xit(j)-1)<10^-5
                        xit(j)=1;
                    end
                end
                % 判定不等于0,1的x的位置
                undef = [];
                for j = 1:scoresik
                    if xit(j) ~= 1
                        if xit(j) ~= 0
                            undef = [undef,j];
                        end
                    end
                end
                noun = length(undef);
                %将已确定的订单插入回本次整数解
                xxi{k} = zeros(1,scoresi); 
                xxi{k}(def{father}) = 1;
                jj = 1;
                for ii = 1:scoresi
                    if xxi{k}(ii) == 0
                        xxi{k}(ii) = xit(jj);
                        jj = jj+1;
                    else
                        xxi{k}(ii) = xdef{k}(def{father}==ii);
                    end
                end
                if f'*xxi{k}' > best
                    issol(k) = 3;
                    continue
                else
                    if noun == 0 %有整数解
                    issol(k) = 1;
                    opti(k) = f'*xxi{k}';
                    best = min(opti);
                    else %无整数解
                        for kk = 1:length(def{father})
                            temp = sort(def{father});
                            if temp(kk) <= undef(1)
                                undef(1) = undef(1)+1;
                            end
                        end
                        def{k} = [def{father},undef(1)];
                    end
                end
            end
            if mod(k,2) == 1 %右儿子，对应让未确定订单取1
                xdef{k} = [xdef{father},1];
                ft = f;
                Vt = V;
                bt = b;
                Ot = O;
                beqt = beq;
                ubt = ub;
                lbt = lb;
                ft(def{father}) = [];
                Vt(:,def{father}) = [];
                Ot(:,def{father}) = [];
                del = def{father}.*xdef{k};
                del(del==0)=[];
                bt = bt-sum(V(:,del),2);
                beqt = beqt-sum(O(:,del),2);
                ubt(def{father}) = [];
                lbt(def{father}) = [];
                xit = linprog(ft,Vt,bt,Ot,beqt,lbt,ubt);
                if isempty(xit) == 1 %本支无解
                    issol(k) = 2;
                    continue
                end
                % 将近似0,1的x更新为0,1
                scoresik = length(xit);
                for j = 1:scoresik
                    if abs(xit(j))<10^-5
                        xit(j)=0;
                    end
                    if abs(xit(j)-1)<10^-5
                        xit(j)=1;
                    end
                end
                % 判定不等于0,1的x的位置
                undef = [];
                for j = 1:scoresik
                    if xit(j) ~= 1
                        if xit(j) ~= 0
                            undef = [undef,j];
                        end
                    end
                end
                noun = length(undef);
                %将已确定的订单插入回本次整数解
                xxi{k} = zeros(1,scoresi); 
                xxi{k}(def{father}) = 1;
                jj = 1;
                for ii = 1:scoresi
                    if xxi{k}(ii) == 0
                        xxi{k}(ii) = xit(jj);
                        jj = jj+1;
                    else
                        xxi{k}(ii) = xdef{k}(def{father}==ii);
                    end
                end
                if f'*xxi{k}' > best
                    issol(k) = 3;
                    continue
                else
                    if noun == 0 %有整数解
                    issol(k) = 1;
                    opti(k) = f'*xxi{k}';
                    best = min(opti);
                    else %无整数解
                        for kk = 1:length(def{father})
                            temp = sort(def{father});
                            if temp(kk) <= undef(1)
                                undef(1) = undef(1)+1;
                            end
                        end
                        def{k} = [def{father},undef(1)];
                    end
                end
            end
        else
            issol(k) = 1;
        end
    end
    opt(i) = min(opti);
    x{i} = xxi(opti==min(opti));
end
    

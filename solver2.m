h = 11; %��֧����
load('data.mat')
x = cell(task,1);
opt = zeros(task,1);
% �ɳڵ�ԭ���������Լ�������LP 
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
    % ������0,1��x����Ϊ0,1
    for j = 1:scoresi
        if abs(xi(j))<10^-5
            xi(j)=0;
        end
        if abs(xi(j)-1)<10^-5
            xi(j)=1;
        end
    end
    % �ж�������0,1��x��λ��
    undef = [];
    for j = 1:scoresi
        if xi(j) ~= 1
            if xi(j) ~= 0
                undef = [undef,j];
            end
        end
    end
    noun = length(undef);
    % �����߶�Ϊh�Ķ�����
    issol = zeros(1,2^h-1); %�Ƿ��н�����0��ʾ�������⣬1��ʾ�������⣬2��ʾ�޽⣬3��ʾ�ⳬ������ֵ�Ͻ�
    def = cell(1,2^h-1); %��ȷ�����������(������������ڵ㸳ֵ)
    xdef = cell(1,2^h-1); %��ȷ������ȡ0����1��(�������ӽڵ㸳ֵ)
    opti = inf * ones(1,2^h-1); %����ֵ��(������������ڵ㸳ֵ)
    xxi = cell(1,2^h-1); %����(���н�ڵ㸳ֵ)
    if noun == 0 %����������
        issol(1) = 1;
        opti(1) = f'*xi;
        xxi{1} = xi;
    else %������������
        def{1} = undef(1); 
    end
    best = inf;
    for k = 2:2^h-1
        father = floor(k/2);
        if issol(father) == 0 %���Ը��ڵ�����������������·�֧
            if mod(k,2) == 0 %����ӣ���Ӧ��δȷ������ȡ0
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
                if isempty(xit) == 1 %��֧�޽�
                    issol(k) = 2;
                    continue
                end
                % ������0,1��x����Ϊ0,1
                scoresik = length(xit);
                for j = 1:scoresik
                    if abs(xit(j))<10^-5
                        xit(j)=0;
                    end
                    if abs(xit(j)-1)<10^-5
                        xit(j)=1;
                    end
                end
                % �ж�������0,1��x��λ��
                undef = [];
                for j = 1:scoresik
                    if xit(j) ~= 1
                        if xit(j) ~= 0
                            undef = [undef,j];
                        end
                    end
                end
                noun = length(undef);
                %����ȷ���Ķ�������ر���������
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
                    if noun == 0 %��������
                    issol(k) = 1;
                    opti(k) = f'*xxi{k}';
                    best = min(opti);
                    else %��������
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
            if mod(k,2) == 1 %�Ҷ��ӣ���Ӧ��δȷ������ȡ1
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
                if isempty(xit) == 1 %��֧�޽�
                    issol(k) = 2;
                    continue
                end
                % ������0,1��x����Ϊ0,1
                scoresik = length(xit);
                for j = 1:scoresik
                    if abs(xit(j))<10^-5
                        xit(j)=0;
                    end
                    if abs(xit(j)-1)<10^-5
                        xit(j)=1;
                    end
                end
                % �ж�������0,1��x��λ��
                undef = [];
                for j = 1:scoresik
                    if xit(j) ~= 1
                        if xit(j) ~= 0
                            undef = [undef,j];
                        end
                    end
                end
                noun = length(undef);
                %����ȷ���Ķ�������ر���������
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
                    if noun == 0 %��������
                    issol(k) = 1;
                    opti(k) = f'*xxi{k}';
                    best = min(opti);
                    else %��������
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
    

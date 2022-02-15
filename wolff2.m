mag= [];
chi = [];
E0 = [];
l =64;
% 模型大小(边长)取值
for T = 0:0.01:5
    beta = 1/T;
    p = 1-exp(-2*beta);
    % 计算每一个温度对应的加入概率
    warm = 10;
    measure = 20;
    % 对此温度区间的warmup次数和measure次数取值
    mag1 = [];
    E1 =[];
    m = ones(l);
    % 初始图取为全部自旋向上的铁磁态
    for i = 1:warm
        x = randi(l);
        y = randi(l);
        % 随机选点
        cluster = [x,y];
        % cluster： 所有要翻转点坐标
        new = [x,y];
        % new： 上一轮加入cluster的点坐标
        newn = 1;
        % newn： 上一轮加入cluster的点个数
        mm = zeros(l);
        mm(x,y) = 1;
        % mm： cluster的图
        while newn > 0
            neww = [];
            % neww： 这一轮加入cluster的点坐标
            for j = 1:newn
                % 对于每一个上一轮加入的坐标，判定上下左右四个邻居
                if new(j,1)+1 <= l 
                    % 判定邻居是否超出边界
                    if mm(new(j,1)+1,new(j,2)) == 0
                        % 判定邻居是否已经在cluster里
                        if m(new(j,1)+1,new(j,2)) == m(new(j,1),new(j,2))
                            % 判定邻居是否与本身自旋相同
                            if rand(1) < p
                                % 取P_add概率
                                neww = [neww;new(j,1)+1,new(j,2)];
                                % 将邻居坐标加入neww
                                mm(new(j,1)+1,new(j,2)) = 1;
                                % 更新cluster的图
                            end
                        end
                    end
                end
                if new(j,1)-1 >= 1 
                    if mm(new(j,1)-1,new(j,2)) == 0
                        if m(new(j,1)-1,new(j,2)) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1)-1,new(j,2)];
                                mm(new(j,1)-1,new(j,2)) = 1;
                            end
                        end
                    end
                end
                if new(j,2)+1 <= l 
                    if mm(new(j,1),new(j,2)+1) == 0
                        if m(new(j,1),new(j,2)+1) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1),new(j,2)+1];
                                mm(new(j,1),new(j,2)+1) = 1;
                            end
                        end
                    end
                end
                if new(j,2)-1 >= 1 
                    if mm(new(j,1),new(j,2)-1) == 0
                        if m(new(j,1),new(j,2)-1) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1),new(j,2)-1];
                                mm(new(j,1),new(j,2)-1) = 1;
                            end
                        end
                    end
                end
            end
            new = neww;
            cluster = [cluster;neww];
            if isempty(neww) == 0
                newn = length(neww(:,1));
            else
                newn = 0;
            end
            % 更新new, cluster, newn
        end
                
        for k = 1:length(cluster(:,1))
            m(cluster(k,1),cluster(k,2)) = 1 - m(cluster(k,1),cluster(k,2));
            % 翻转cluster中每一个坐标对应的点
        end
    end
    for i = 1:measure
        x = randi(l);
        y = randi(l);
        cluster = [x,y];
        new = [x,y];
        newn = 1;
        mm = zeros(l);
        mm(x,y) = 1;
        while newn > 0
            neww = [];
            for j = 1:newn
                if new(j,1)+1 <= l 
                    if mm(new(j,1)+1,new(j,2)) == 0
                        if m(new(j,1)+1,new(j,2)) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1)+1,new(j,2)];
                                mm(new(j,1)+1,new(j,2)) = 1;
                            end
                        end
                    end
                end
                if new(j,1)-1 >= 1 
                    if mm(new(j,1)-1,new(j,2)) == 0
                        if m(new(j,1)-1,new(j,2)) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1)-1,new(j,2)];
                                mm(new(j,1)-1,new(j,2)) = 1;
                            end
                        end
                    end
                end
                if new(j,2)+1 <= l 
                    if mm(new(j,1),new(j,2)+1) == 0
                        if m(new(j,1),new(j,2)+1) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1),new(j,2)+1];
                                mm(new(j,1),new(j,2)+1) = 1;
                            end
                        end
                    end
                end
                if new(j,2)-1 >= 1 
                    if mm(new(j,1),new(j,2)-1) == 0
                        if m(new(j,1),new(j,2)-1) == m(new(j,1),new(j,2))
                            if rand(1) < p
                                neww = [neww;new(j,1),new(j,2)-1];
                                mm(new(j,1),new(j,2)-1) = 1;
                            end
                        end
                    end
                end
            end
            new = neww;
            cluster = [cluster;neww];
            if isempty(neww) == 0
                newn = length(neww(:,1));
            else
                newn = 0;
            end
        end
                
        for k = 1:length(cluster(:,1))
            m(cluster(k,1),cluster(k,2)) = 1 - m(cluster(k,1),cluster(k,2));
        end
        mag11 = abs(2*sum(sum(m))-(l*l))/(l*l);
        mag1 = [mag1,mag11];
        % 计算磁化强度mag并储存
        E = 0;
        for jx = 1:l
            for jy = 1:l
                if jx < l
                    if m(jx+1,jy) == m(jx,jy)
                        E = E-1;
                    else
                        E = E+1;
                    end
                end
                if jy < l
                    if m(jx,jy+1) == m(jx,jy)
                        E = E-1;
                    else
                        E = E+1;
                    end
                end
            end
        end
        Emax = l*(l-1)*2;
        E = E/Emax;
        E1 = [E1,E];
        % 对于任意两个相邻电子，计算其作用能并累加
    end
    mag = [mag,sum(mag1)/measure];
    chi = [chi,(sum(mag1 .* mag1) - (sum(mag1)*sum(mag1)/measure))/T];
    E0 = [E0,sum(E1)/measure];
    % 计算该温度下平均磁化强度，磁化率，平均能量
end

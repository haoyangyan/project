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
        % 随机取点
        m1 = zeros(l);
        m2 = zeros(l);
        m1(x,y) = 1;
        % m1为cluster里所有点的图
        m2(x,y) = 1;
        % m2为上一轮加入的点的图
        len = 1;
        % len： 上一轮加入点个数
        while len>0
            m3 = zeros(l);
            for j = 1:len
                % 对于每个上一轮加入点，判定上下左右四个邻居
                if x(j)+1 <= l
                    % 判定邻居是否超出边界
                    if m(x(j),y(j)) == m(x(j)+1,y(j))
                        % 判定邻居是否与本身自旋相同
                        m3(x(j)+1,y(j)) = 1;
                        % 将邻居点画到图m3中
                    end
                end
                if x(j)-1 >= 1
                    if m(x(j),y(j)) == m(x(j)-1,y(j))
                        m3(x(j)-1,y(j)) = 1;
                    end
                end
                if y(j)+1 <= l
                    if m(x(j),y(j)) == m(x(j),y(j)+1)
                        m3(x(j),y(j)+1) = 1;
                    end
                end
                if y(j)-1 >= 1
                    if m(x(j),y(j)) == m(x(j),y(j)-1)
                        m3(x(j),y(j)-1) = 1;
                    end
                end
            end
            m4 = m3 - m1 - rand(l)/p;
            % 图m4为m3(这一轮新加入所有邻居，可能包含这一轮前就有的点)减去m1(这一轮前的点)减去在[0,1/p]区间随机矩阵
            m4(m4 > 0) = 1;
            m4(m4 <= 0) = 0;
            % 如果m4>0则保留此邻居，反之则不保留，得到的就是这一轮要加入cluster的点
            m2 = m4;
            % 更新上一轮加入点的图
            [x,y] = find(m2==1);
            len = length(x);
            % 查找上一轮加入点的坐标并计算个数
            m1 = m4 + m1;
            % 更新整个cluster的图
        end
        m = m1-m;
        m(m == -1) = 1;
        % 将cluster的图减去原图再取绝对值，使得cluster内的点0,1互换
    end
    for i = 1:measure
        % 再做measure次相同操作
        x = randi(l);
        y = randi(l);
        m1 = zeros(l);
        m2 = zeros(l);
        m1(x,y) = 1;
        m2(x,y) = 1;
        len = 1;
        while len>0
            m3 = zeros(l);
            for j = 1:len
                if x(j)+1 <= l
                    if m(x(j),y(j)) == m(x(j)+1,y(j))
                        m3(x(j)+1,y(j)) = 1;
                    end
                end
                if x(j)-1 >= 1
                    if m(x(j),y(j)) == m(x(j)-1,y(j))
                        m3(x(j)-1,y(j)) = 1;
                    end
                end
                if y(j)+1 <= l
                    if m(x(j),y(j)) == m(x(j),y(j)+1)
                        m3(x(j),y(j)+1) = 1;
                    end
                end
                if y(j)-1 >= 1
                    if m(x(j),y(j)) == m(x(j),y(j)-1)
                        m3(x(j),y(j)-1) = 1;
                    end
                end
            end
            m4 = m3 - m1 - rand(l)/p;
            m4(m4 > 0) = 1;
            m4(m4 <= 0) = 0;
            m2 = m4;
            [x,y] = find(m2==1);
            len = length(x);
            m1 = m4 + m1;
        end
        m = m1-m;
        m(m == -1) = 1;
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

                    
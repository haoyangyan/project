mag= [];
chi = [];
E0 = [];
l =64;
% 模型大小(边长)取值
for T = 0:0.01:5
    beta = 1/T;
    p = 1-exp(-2*beta);
    % 计算每一个温度对应的加入概率
    warm = 200000;
    measure = 10000;
    % 对此温度区间的warmup次数和measure次数取值
    mag1 = [];
    E1 =[];
    m = ones(l);
    % 初始图取为全部自旋向上的铁磁态
    for i = 1:warm
        x = randi(l);
        y = randi(l);

        ei = 0;
        if y+1 <= l
            if m(x,y+1) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if y-1 >= 1
            if m(x,y-1) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if x+1 <= l 
            if m(x+1,y) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if x-1 >= 1
            if m(x-1,y) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        
        ej = 0;
        if y+1 <= l
            if m(x,y+1) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if y-1 >= 1
            if m(x,y-1) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if x+1 <= l 
            if m(x+1,y) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if x-1 >= 1
            if m(x-1,y) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end

        de = ej - ei;
        if de <= 0
            if m(x,y) == 0
                m(x,y) = 1;
            else
                m(x,y) = 0;
            end
        end
        if de > 0
            rho = exp(-beta*de);
            r = rand(1);
            if r <= rho
                if m(x,y) == 0
                    m(x,y) = 1;
                else
                    m(x,y) = 0;
                end
            end
        end
    end
    for i = 1:measure
        x = randi(l);
        y = randi(l);

        ei = 0;
        if y+1 <= l
            if m(x,y+1) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if y-1 >= 1
            if m(x,y-1) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if x+1 <= l 
            if m(x+1,y) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
        if x-1 >= 1
            if m(x-1,y) == m(x,y)
                ei = ei-1;
            else
                ei = ei+1;
            end
        end
              
        ej = 0;
        if y+1 <= l
            if m(x,y+1) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if y-1 >= 1
            if m(x,y-1) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if x+1 <= l 
            if m(x+1,y) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end
        if x-1 >= 1
            if m(x-1,y) == m(x,y)
                ej = ej+1;
            else
                ej = ej-1;
            end
        end

        de = ej - ei;
        if de <= 0
            if m(x,y) == 0
                m(x,y) = 1;
            else
                m(x,y) = 0;
            end
        end
        if de > 0
            rho = exp(-beta*de);
            r = rand(1);
            if r <= rho
                if m(x,y) == 0
                    m(x,y) = 1;
                else
                    m(x,y) = 0;
                end
            end
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
mag= [];
chi = [];
E0 = [];
l =64;
% ģ�ʹ�С(�߳�)ȡֵ
for T = 0:0.01:5
    beta = 1/T;
    p = 1-exp(-2*beta);
    % ����ÿһ���¶ȶ�Ӧ�ļ������
    warm = 10;
    measure = 20;
    % �Դ��¶������warmup������measure����ȡֵ
    mag1 = [];
    E1 =[];
    m = ones(l);
    % ��ʼͼȡΪȫ���������ϵ�����̬
    for i = 1:warm
        x = randi(l);
        y = randi(l);
        % ���ѡ��
        cluster = [x,y];
        % cluster�� ����Ҫ��ת������
        new = [x,y];
        % new�� ��һ�ּ���cluster�ĵ�����
        newn = 1;
        % newn�� ��һ�ּ���cluster�ĵ����
        mm = zeros(l);
        mm(x,y) = 1;
        % mm�� cluster��ͼ
        while newn > 0
            neww = [];
            % neww�� ��һ�ּ���cluster�ĵ�����
            for j = 1:newn
                % ����ÿһ����һ�ּ�������꣬�ж����������ĸ��ھ�
                if new(j,1)+1 <= l 
                    % �ж��ھ��Ƿ񳬳��߽�
                    if mm(new(j,1)+1,new(j,2)) == 0
                        % �ж��ھ��Ƿ��Ѿ���cluster��
                        if m(new(j,1)+1,new(j,2)) == m(new(j,1),new(j,2))
                            % �ж��ھ��Ƿ��뱾��������ͬ
                            if rand(1) < p
                                % ȡP_add����
                                neww = [neww;new(j,1)+1,new(j,2)];
                                % ���ھ��������neww
                                mm(new(j,1)+1,new(j,2)) = 1;
                                % ����cluster��ͼ
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
            % ����new, cluster, newn
        end
                
        for k = 1:length(cluster(:,1))
            m(cluster(k,1),cluster(k,2)) = 1 - m(cluster(k,1),cluster(k,2));
            % ��תcluster��ÿһ�������Ӧ�ĵ�
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
        % ����Ż�ǿ��mag������
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
        % ���������������ڵ��ӣ������������ܲ��ۼ�
    end
    mag = [mag,sum(mag1)/measure];
    chi = [chi,(sum(mag1 .* mag1) - (sum(mag1)*sum(mag1)/measure))/T];
    E0 = [E0,sum(E1)/measure];
    % ������¶���ƽ���Ż�ǿ�ȣ��Ż��ʣ�ƽ������
end

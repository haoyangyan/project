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
        % ���ȡ��
        m1 = zeros(l);
        m2 = zeros(l);
        m1(x,y) = 1;
        % m1Ϊcluster�����е��ͼ
        m2(x,y) = 1;
        % m2Ϊ��һ�ּ���ĵ��ͼ
        len = 1;
        % len�� ��һ�ּ�������
        while len>0
            m3 = zeros(l);
            for j = 1:len
                % ����ÿ����һ�ּ���㣬�ж����������ĸ��ھ�
                if x(j)+1 <= l
                    % �ж��ھ��Ƿ񳬳��߽�
                    if m(x(j),y(j)) == m(x(j)+1,y(j))
                        % �ж��ھ��Ƿ��뱾��������ͬ
                        m3(x(j)+1,y(j)) = 1;
                        % ���ھӵ㻭��ͼm3��
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
            % ͼm4Ϊm3(��һ���¼��������ھӣ����ܰ�����һ��ǰ���еĵ�)��ȥm1(��һ��ǰ�ĵ�)��ȥ��[0,1/p]�����������
            m4(m4 > 0) = 1;
            m4(m4 <= 0) = 0;
            % ���m4>0�������ھӣ���֮�򲻱������õ��ľ�����һ��Ҫ����cluster�ĵ�
            m2 = m4;
            % ������һ�ּ�����ͼ
            [x,y] = find(m2==1);
            len = length(x);
            % ������һ�ּ��������겢�������
            m1 = m4 + m1;
            % ��������cluster��ͼ
        end
        m = m1-m;
        m(m == -1) = 1;
        % ��cluster��ͼ��ȥԭͼ��ȡ����ֵ��ʹ��cluster�ڵĵ�0,1����
    end
    for i = 1:measure
        % ����measure����ͬ����
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

                    
Game = inputdlg('输入比赛名称 Enter the game name','Game');
Gamen = Game{1};
TMcell = inputdlg('输入所有队员名字及号码并以空格分隔 Enter all the team members splited with space           示例：“张三1 李四2 王五3”','TeamMember');
TMstring = string(TMcell);
TMstringN = regexp(TMstring,' ','split');
TM = cellstr(TMstringN);
%将队员名字及号码储存在1*N的cell中，相应编号1-N，命名为TM
NTM = length(TM);
NB = [];
for i = 1:NTM
    yy = isstrprop(TM{i},'digit');
    xx = TM{i}(yy);
    NBI = str2num(xx);
    NB = [NB,NBI];
end
%将队员号码储存为1*N的矩阵，命名为NB

alldata = {};

for j = 1:100
    OK = 0;
    while OK == 0
        [OD,OK] = listdlg('ListString',{'进攻 Offense','防守 Defense'},'Name','OD','PromptString','选择这一分是进攻还是防守 Is this point offense or defense','SelectionMode','Single');
    end
    %将进攻或防守储存为值，进攻为1，防守为2，命名为OD

    OK = 0;
    while OK == 0
        [Player,OK] = listdlg('ListString',TM,'Name','Player','PromptString','选择这一分上场队员 Who play this point','SelectionMode','Multiple');
    end
    %将这一分的上场队员的编号储存为1*7的矩阵，命名为Player

    PN = cell(1,7);
    for i = 1:7
        PN{i} = TM{Player(i)};
    end
    %将这一分的上场队员名字及号码储存为1*7的cell，命名为PN
    
    PNB = [];
    for i = 1:7
        PNB = [PNB,NB(Player(i))];
    end
    %将这一分的上场队员号码储存为1*7的cell，命名为PNB

    AA = {'P pass 直传','S swing 横传','D dump 短盘','H huck 长盘','U upline 给控盘手向前推进传盘','B break 破防','G goal 得分','A stallout 超时','R drop 接盘失误'};
    %将接盘人做了什么储存为1*9的cell，命名为AA
    AAA = {'P','S','D','H','U','B','G','A','R'};
    BB = {PN{1},PN{2},PN{3},PN{4},PN{5},PN{6},PN{7},'T throwaway 传盘出界/落地','K blocked 被抄截','L duel 争盘'};
    %将盘传给了谁储存为1*10的cell，命名为BB
    BBB = {PNB(1),PNB(2),PNB(3),PNB(4),PNB(5),PNB(6),PNB(7),'T','K','L'};
    CC = {PN{1},PN{2},PN{3},PN{4},PN{5},PN{6},PN{7},'T throwaway 盘出界/落地','K blocked 被抄截'};
    %将duel的结果储存为1*9的cell，命名为CC
    CCC = {PNB(1),PNB(2),PNB(3),PNB(4),PNB(5),PNB(6),PNB(7),'T','K'};
    DD = {'K block 对方被抄截','T throwaway 对方盘出界/落地','G goal 对方得分'};
    %将攻防转换后的结果储存为1*3的cell，命名为DD
    DDD = {'K','T','G'};

    data = {};

    if OD == 1
        %进攻分的过程
        OK = 0;
        while OK == 0
            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
        end
        data = [data,PNB(Who)];
        %将一个人做了什么到下一个人做了什么过程设为一个A循环，如下所示
        for i = 1:10000
            OK = 0;
            while OK == 0
                [a,OK] = listdlg('ListString',AA,'Name','Do','PromptString','接盘人做了什么 What does he/she do','SelectionMode','Single');
            end
            data = [data,AAA(a)];

            if 1<=a && a<=6
                OK = 0;
                while OK == 0
                    [b,OK] = listdlg('ListString',BB,'Name','Who','PromptString','将盘传给谁 Who the disc passes to','SelectionMode','Single');
                end

                if 1<=b && b<=7
                    data = [data,BBB(b)];
                end
                if 8<=b && b<=9
                    data = [data,BBB(b)];
                    OK = 0;
                    while OK == 0
                        [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                    end
                    data = [data,DDD(d)];
                    if d == 1
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d == 2
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d ==3
                        break
                    end
                end
                if b == 10
                    data = [data,BBB(b)];
                    OK = 0;
                    while OK == 0
                        [c,OK] = listdlg('ListString',CC,'Name','Result','PromptString','争盘结果如何 what is the result of duel ','SelectionMode','Single');
                    end
                    data = [data,CCC(c)];
                    if 1<=c && c<=7
                    end
                    if 8<=c && c<=9
                        OK = 0;
                        while OK == 0
                            [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                        end
                        data = [data,DDD(d)];
                        if d == 1
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d == 2
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d ==3
                            break
                        end
                    end
                end
            end

            if a == 7
                break
            end

            if 8<=a && a<=9
                OK = 0;
                while OK == 0
                    [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                end
                data = [data,DDD(d)];
                if d == 1
                    OK = 0;
                    while OK == 0
                        [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                    end
                    data = [data,PNB(Who)];
                    OK = 0;
                    while OK == 0
                        [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                    end
                    data = [data,PNB(Who)];
                end
                if d == 2
                    OK = 0;
                    while OK == 0
                        [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                    end
                    data = [data,PNB(Who)];
                end
                if d ==3
                    break
                end
            end
        end
        %A循环结束
    end

    
    if OD ==2
        %防守分的过程
        OK = 0;
        while OK == 0
            [e,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
        end
        data = [data,DDD(e)];

        if e == 1
            OK = 0;
            while OK == 0
                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
            end
            data = [data,PNB(Who)];
            OK = 0;
            while OK == 0
                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
            end
            data = [data,PNB(Who)];
            %调用A循环
            for i = 1:10000
                OK = 0;
                while OK == 0
                    [a,OK] = listdlg('ListString',AA,'Name','Do','PromptString','接盘人做了什么 What does he/she do','SelectionMode','Single');
                end
                data = [data,AAA(a)];

                if 1<=a && a<=6
                    OK = 0;
                    while OK == 0
                        [b,OK] = listdlg('ListString',BB,'Name','Who','PromptString','将盘传给谁 Who the disc passes to','SelectionMode','Single');
                    end

                    if 1<=b && b<=7
                        data = [data,BBB(b)];
                    end
                    if 8<=b && b<=9
                        data = [data,BBB(b)];
                        OK = 0;
                        while OK == 0
                            [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                        end
                        data = [data,DDD(d)];
                        if d == 1
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d == 2
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d ==3
                            break
                        end
                    end
                    if b == 10
                        data = [data,BBB(b)];
                        OK = 0;
                        while OK == 0
                            [c,OK] = listdlg('ListString',CC,'Name','Result','PromptString','争盘结果如何 what is the result of duel ','SelectionMode','Single');
                        end
                        data = [data,CCC(c)];
                        if 1<=c && c<=7
                        end
                        if 8<=c && c<=9
                            OK = 0;
                            while OK == 0
                                [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                            end
                            data = [data,DDD(d)];
                            if d == 1
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                            end
                            if d == 2
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                            end
                            if d ==3
                                break
                            end
                        end
                    end
                end

                if a == 7
                    break
                end

                if 8<=a && a<=9
                    OK = 0;
                    while OK == 0
                        [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                    end
                    data = [data,DDD(d)];
                    if d == 1
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d == 2
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d ==3
                        break
                    end
                end
            end
            %A循环结束
        end

        if e == 2
            OK = 0;
            while OK == 0
                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
            end
            data = [data,PNB(Who)];
            %调用A循环
            for i = 1:10000
                OK = 0;
                while OK == 0
                    [a,OK] = listdlg('ListString',AA,'Name','Do','PromptString','接盘人做了什么 What does he/she do','SelectionMode','Single');
                end
                data = [data,AAA(a)];

                if 1<=a && a<=6
                    OK = 0;
                    while OK == 0
                        [b,OK] = listdlg('ListString',BB,'Name','Who','PromptString','将盘传给谁 Who the disc passes to','SelectionMode','Single');
                    end

                    if 1<=b && b<=7
                        data = [data,BBB(b)];
                    end
                    if 8<=b && b<=9
                        data = [data,BBB(b)];
                        OK = 0;
                        while OK == 0
                            [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                        end
                        data = [data,DDD(d)];
                        if d == 1
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d == 2
                            OK = 0;
                            while OK == 0
                                [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                            end
                            data = [data,PNB(Who)];
                        end
                        if d ==3
                            break
                        end
                    end
                    if b == 10
                        data = [data,BBB(b)];
                        OK = 0;
                        while OK == 0
                            [c,OK] = listdlg('ListString',CC,'Name','Result','PromptString','争盘结果如何 what is the result of duel ','SelectionMode','Single');
                        end
                        data = [data,CCC(c)];
                        if 1<=c && c<=7
                        end
                        if 8<=c && c<=9
                            OK = 0;
                            while OK == 0
                                [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                            end
                            data = [data,DDD(d)];
                            if d == 1
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                            end
                            if d == 2
                                OK = 0;
                                while OK == 0
                                    [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                                end
                                data = [data,PNB(Who)];
                            end
                            if d ==3
                                break
                            end
                        end
                    end
                end

                if a == 7
                    break
                end

                if 8<=a && a<=9
                    OK = 0;
                    while OK == 0
                        [d,OK] = listdlg('ListString',DD,'Name','Turnover','PromptString','对方做了什么 What does the opponent do','SelectionMode','Single');
                    end
                    data = [data,DDD(d)];
                    if d == 1
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','谁抄截了对方 Who block the opponent','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d == 2
                        OK = 0;
                        while OK == 0
                            [Who,OK] = listdlg('ListString',PN,'Name','Who','PromptString','盘在谁手中 Who get the disc','SelectionMode','Single');
                        end
                        data = [data,PNB(Who)];
                    end
                    if d ==3
                        break
                    end
                end
            end
            %A循环结束
        end

        if e ==3
        end
    end

    alldata = [alldata;{OD,PNB,data}];
    %将这一分的 进攻/防守 上场队员编号 发生事件 储存进alldata中，为m*3的cell，其中m为进行分数

    END = {'进入下一分 to next point','比赛结束了 game is over'};
    OK = 0;
    while OK == 0
        [END,OK] = listdlg('ListString',END,'Name','END','PromptString','比赛结束了吗 is game over','SelectionMode','Single');
    end
    if END == 2
        break
    end
end


[Points,Len] = size(alldata);
%比赛总共打了Points分

Nstat = 40;
%总共Nstat-1类数据

stat = cell(NTM+2,Nstat);
%总数据表格为NTM+2*Nstat的cell

stat{2,1} = 'team';
for i = 1:NTM
    stat{i+2,1} = TM{i};
end
for i = 2:NTM+2
    for j = 2:19
        stat{i,j} = 0;
    end
    for j = 21:27
        stat{i,j} = 0;
    end
    for j = 29:40
        stat{i,j} = 0;
    end
end

stat{1,2} = '上场分数pointsplayed';
for i = 1:Points
    for j = 1:7
        tep2 = find(NB == alldata{i,2}(j));
        stat{tep2+2,2} = stat{tep2+2,2}+1;
    end
end
stat{2,2} = sum(cell2mat(stat(3:NTM+2,2)))/7;

stat{1,3} = '进攻分OPP';
for i = 1:Points
    if alldata{i,1} == 1
         for j = 1:7
            tep3 = find(NB == alldata{i,2}(j));
            stat{tep3+2,3} = stat{tep3+2,3}+1;
        end
    end
end
stat{2,3} = sum(cell2mat(stat(3:NTM+2,3)))/7;

stat{1,4} = '防守分DPP';
for i = 1:Points
    if alldata{i,1} == 2
         for j = 1:7
            tep4 = find(NB == alldata{i,2}(j));
            stat{tep4+2,4} = stat{tep4+2,4}+1;
        end
    end
end
stat{2,4} = sum(cell2mat(stat(3:NTM+2,4)))/7;

stat{1,5} = '得分goals';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 1
                    if isa(alldata{i,3}{j-1},'double')
                        tep5 = find(NB == alldata{i,3}{j-1});
                        stat{tep5+2,5} = stat{tep5+2,5}+1;
                    end
                end
            end
        end
    end
end
stat{2,5} = sum(cell2mat(stat(3:NTM+2,5)));

stat{1,6} = '助攻assists';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 2
                    if isa(alldata{i,3}{j-1},'double')
                        tep = alldata{i,3}{j-2};
                        if isa(tep,'char')
                            if tep=='P'|| tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                                if j > 3
                                    tep6 = find(NB == alldata{i,3}{j-3});
                                    stat{tep6+2,6} = stat{tep6+2,6}+1;
                                end
                            end
                            if tep == 'L'
                                if j > 4
                                    tep6 = find(NB == alldata{i,3}{j-4});
                                    stat{tep6+2,6} = stat{tep6+2,6}+1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,6} = sum(cell2mat(stat(3:NTM+2,6)));

stat{1,7} = '抄截Blocks';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'K'
                if isa(alldata{i,3}{j+1},'double')
                    tep7 = find(NB == alldata{i,3}{j+1});
                    stat{tep7+2,7} = stat{tep7+2,7}+1;
                end
            end
        end
    end
end
stat{2,7} = sum(cell2mat(stat(3:NTM+2,7)));

stat{1,8} = '成功传盘completions';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                if j+2 <= l
                    if isa(alldata{i,3}{j+1},'double')
                        if isa(alldata{i,3}{j+2},'char')
                            if alldata{i,3}{j+2} ~= 'R'
                                tep8 = find(NB == alldata{i,3}{j-1});
                                stat{tep8+2,8} = stat{tep8+2,8}+1;
                            end
                        end
                    end
                    if isa(alldata{i,3}{j+1},'char')
                        if alldata{i,3}{j+1} == 'L'
                            if isa(alldata{i,3}{j+2},'double')
                                tep8 = find(NB == alldata{i,3}{j-1});
                                stat{tep8+2,8} = stat{tep8+2,8}+1;
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,8} = sum(cell2mat(stat(3:NTM+2,8)));

draft = cell(NTM+2,17);
draft{2,1} = 'team';
for i = 1:NTM
    draft{i+2,1} = TM{i};
end
for i = 2:NTM+2
    for j = 2:17
        draft{i,j} = 0;
    end
end

draft{1,2} = '总传盘';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                tep02 = find(NB == alldata{i,3}{j-1});
                draft{tep02+2,2} = draft{tep02+2,2}+1;
            end
        end
    end
end
draft{2,2} = sum(cell2mat(draft(3:NTM+2,2)));

stat{1,9} = '传盘成功率CMP%';
for i = 2:NTM+2
    stat{i,9} = stat{i,8} / draft{i,2} * 100;
end

stat{1,10} = '接盘失误drops';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'R'
                if isa(alldata{i,3}{j-1},'double')
                    tep10 = find(NB == alldata{i,3}{j-1});
                    stat{tep10+2,10} = stat{tep10+2,10}+1;
                end
            end
        end
    end
end
stat{2,10} = sum(cell2mat(stat(3:NTM+2,10)));

draft{1,3} = '传盘出界/落地throwaways';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                if j+2 <= l
                    if isa(alldata{i,3}{j+1},'char')
                        if alldata{i,3}{j+1} == 'T'
                            tep03 = find(NB == alldata{i,3}{j-1});
                            draft{tep03+2,3} = draft{tep03+2,3}+1;
                        end
                        if alldata{i,3}{j+1} == 'L'
                            if isa(alldata{i,3}{j+2},'char')
                                if alldata{i,3}{j+2} == 'T'
                                    tep03 = find(NB == alldata{i,3}{j-1});
                                    draft{tep03+2,3} = draft{tep03+2,3}+1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
draft{2,3} = sum(cell2mat(draft(3:NTM+2,3)));

draft{1,4} = '传盘被抄截/blocked';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                if j+2 <= l
                    if isa(alldata{i,3}{j+1},'char')
                        if j > 1
                            if alldata{i,3}{j+1} == 'K'
                                tep04 = find(NB == alldata{i,3}{j-1});
                                draft{tep04+2,4} = draft{tep04+2,4}+1;
                            end
                            if alldata{i,3}{j+1} == 'L'
                                if isa(alldata{i,3}{j+2},'char')
                                    if alldata{i,3}{j+2} == 'K'
                                        tep04 = find(NB == alldata{i,3}{j-1});
                                        draft{tep04+2,4} = draft{tep04+2,4}+1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
draft{2,4} = sum(cell2mat(draft(3:NTM+2,4)));

stat{1,11} = '传盘失误TRAW/BKed';
for i = 2:NTM+2
    stat{i,11} = draft{i,3} + draft{i,4};
end

stat{1,12} = '传盘出界/落地率TRAW%';
for i = 2:NTM+2
    stat{i,12} = draft{i,3} / draft{i,2};
end

stat{1,13} = '传盘被抄截率BKed%';
for i = 2:NTM+2
    stat{i,13} = draft{i,4} / draft{i,2};
end

stat{1,14} = '被计数停止stalls';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'A'
                if isa(alldata{i,3}{j-1},'double')
                    tep14 = find(NB == alldata{i,3}{j-1});
                    stat{tep14+2,14} = stat{tep14+2,14}+1;
                end
            end
        end
    end
end
stat{2,14} = sum(cell2mat(stat(3:NTM+2,14)));

stat{1,15} = '卡拉汉callahan';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 3
                    if isa(alldata{i,3}{j-1},'double')
                        if isa(alldata{i,3}{j-2},'double')
                            if isa(alldata{i,3}{j-3},'char')
                                if alldata{i,3}{j-3} == 'K'
                                    tep15 = find(NB == alldata{i,3}{j-1});
                                    stat{tep15+2,15} = stat{tep15+2,15}+1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,15} = sum(cell2mat(stat(3:NTM+2,15)));

stat{1,16} = '助攻前一次传盘hockeyassists';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 2
                    if isa(alldata{i,3}{j-1},'double')
                        tep = alldata{i,3}{j-2};
                        if isa(tep,'char')
                            if tep=='P'|| tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                                if j > 5
                                    tep = alldata{i,3}{j-4};
                                    if isa(tep,'char')
                                        if tep=='P'|| tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                                            tep16 = find(NB == alldata{i,3}{j-5});
                                            stat{tep16+2,16} = stat{tep16+2,16}+1;
                                        end
                                        if tep == 'L'
                                            if j > 6
                                                tep16 = find(NB == alldata{i,3}{j-6});
                                                stat{tep16+2,16} = stat{tep16+2,16}+1;
                                            end
                                        end
                                    end
                                end
                            end
                            if tep == 'L'
                                if j > 6
                                    tep = alldata{i,3}{j-5};
                                    if isa(tep,'char')
                                        if tep=='P'|| tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                                            tep16 = find(NB == alldata{i,3}{j-6});
                                            stat{tep16+2,16} = stat{tep16+2,16}+1;
                                        end
                                        if tep == 'L'
                                            if j > 7
                                                tep16 = find(NB == alldata{i,3}{j-7});
                                                stat{tep16+2,16} = stat{tep16+2,16}+1;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,16} = sum(cell2mat(stat(3:NTM+2,16)));

stat{1,17} = '正负值+/-';
for i = 2:NTM+2
    stat{i,17} = stat{i,5} + stat{i,6} + stat{i,7} - stat{i,10} - stat{i,11} - stat{i,14};
end

draft{1,5} = '在场得分数';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 1
                    if isa(alldata{i,3}{j-1},'double')
                        for k = 1:7
                            tep05 = find(NB == alldata{i,2}(k));
                            draft{tep05+2,5} = draft{tep05+2,5}+1;
                        end
                    end
                end
            end
        end
    end
end
draft{2,5} = sum(cell2mat(draft(3:NTM+2,5)))/7;

draft{1,6} = '在场失分数';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'G'
                if j > 1
                    if isa(alldata{i,3}{j-1},'char')
                        for k = 1:7
                            tep06 = find(NB == alldata{i,2}(k));
                            draft{tep06+2,6} = draft{tep06+2,6}+1;
                        end
                    end
                end
            end
        end
    end
end
draft{2,6} = sum(cell2mat(draft(3:NTM+2,6)))/7;

draft{1,7} = '在场进攻回合数';
draft{1,8} = '在场防守回合数';
for i = 1:Points
    l = length(alldata{i,3});
    ii = 0;
    if alldata{i,1} == 1
        for k = 1:7
            tep07 = find(NB == alldata{i,2}(k));
            draft{tep07+2,7} = draft{tep07+2,7}+1;
        end
        for j = 1:l
            tep = alldata{i,3}{j};
            if isa(tep,'char')
                if tep=='A'|| tep=='R'||tep=='T'||tep=='K'
                    ii = ii + 1;
                    if rem(ii,2) == 1
                        for k = 1:7
                            tep08 = find(NB == alldata{i,2}(k));
                            draft{tep08+2,8} = draft{tep08+2,8}+1;
                        end
                    end
                    if rem(ii,2) == 0
                        for k = 1:7
                            tep07 = find(NB == alldata{i,2}(k));
                            draft{tep07+2,7} = draft{tep07+2,7}+1;
                        end
                    end
                end
            end
        end
    end
    if alldata{i,1} == 2
        for k = 1:7
            tep08 = find(NB == alldata{i,2}(k));
            draft{tep08+2,8} = draft{tep08+2,8}+1;
        end
        for j = 1:l
            tep = alldata{i,3}{j};
            if isa(tep,'char')
                if tep=='A'|| tep=='R'||tep=='T'||tep=='K'
                    ii = ii + 1;
                    if rem(ii,2) == 0
                        for k = 1:7
                            tep08 = find(NB == alldata{i,2}(k));
                            draft{tep08+2,8} = draft{tep08+2,8}+1;
                        end
                    end
                    if rem(ii,2) == 1
                        for k = 1:7
                            tep07 = find(NB == alldata{i,2}(k));
                            draft{tep07+2,7} = draft{tep07+2,7}+1;
                        end
                    end
                end
            end
        end
    end
end
draft{2,7} = sum(cell2mat(draft(3:NTM+2,7)))/7;
draft{2,8} = sum(cell2mat(draft(3:NTM+2,8)))/7;

stat{1,18} = '进攻效率OEFF';
for i = 2:NTM+2
    stat{i,18} = draft{i,5} / draft{i,7};
end

stat{1,19} = '防守效率DEFF';
for i = 2:NTM+2
    stat{i,19} = - draft{i,6} / draft{i,8};
end

draft{1,9} = '好盘数';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 2:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                if isa(alldata{i,3}{j+1},'double')                    
                    tep09 = find(NB == alldata{i,3}{j-1});
                    draft{tep09+2,9} = draft{tep09+2,9}+1;
                end
            end
        end
    end
end
draft{2,9} = sum(cell2mat(draft(3:NTM+2,9)));

stat{1,21} = '好盘率%';
for i = 2:NTM+2
    stat{i,21} = draft{i,9} / draft{i,2} * 100;
end

draft{1,10} = '坏盘数';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 2:l
        tep = alldata{i,3}{j};
        if isa(tep,'char')
            if tep=='P'||tep=='S'||tep=='D'||tep=='H'||tep=='U'||tep=='B'
                if isa(alldata{i,3}{j+1},'char')
                    if alldata{i,3}{j+1} == 'T' || alldata{i,3}{j+1} == 'K'
                        tep010 = find(NB == alldata{i,3}{j-1});
                        draft{tep010+2,10} = draft{tep010+2,10}+1;
                    end
                end
            end
        end
    end
end
draft{2,10} = sum(cell2mat(draft(3:NTM+2,10)));

stat{1,22} = '坏盘率%';
for i = 2:NTM+2
    stat{i,22} = draft{i,10} / draft{i,2} * 100;
end

stat{1,23} = '争盘成功';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'L'
                if isa(alldata{i,3}{j+1},'double')
                    tep23 = find(NB == alldata{i,3}{j+1});
                    stat{tep23+2,23} = stat{tep23+2,23}+1;
                end
            end
        end
    end
end
stat{2,23} = sum(cell2mat(stat(3:NTM+2,23)));

stat{1,24} = '下底成功';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'H'
                if j+1 <= l
                    if isa(alldata{i,3}{j+1},'double')
                        tep24 = find(NB == alldata{i,3}{j+1});
                        stat{tep24+2,24} = stat{tep24+2,24}+1;
                    end
                    if isa(alldata{i,3}{j+1},'char')
                        if alldata{i,3}{j+1} == 'L'
                            if isa(alldata{i,3}{j+2},'double')
                                tep24 = find(NB == alldata{i,3}{j+2});
                                stat{tep24+2,24} = stat{tep24+2,24}+1;
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,24} = sum(cell2mat(stat(3:NTM+2,24)));

stat{1,25} = '长传成功';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'H'
                if j+1 <= l
                    if isa(alldata{i,3}{j+1},'double')
                        tep25 = find(NB == alldata{i,3}{j-1});
                        stat{tep25+2,25} = stat{tep25+2,25}+1;
                    end
                    if isa(alldata{i,3}{j+1},'char')
                        if alldata{i,3}{j+1} == 'L'
                            if isa(alldata{i,3}{j+2},'double')
                                tep25 = find(NB == alldata{i,3}{j-1});
                                stat{tep25+2,25} = stat{tep25+2,25}+1;
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,25} = sum(cell2mat(stat(3:NTM+2,25)));
              
stat{1,26} = '破防成功';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'B'
                if j+1 <= l
                    if isa(alldata{i,3}{j+1},'double')
                        tep26 = find(NB == alldata{i,3}{j-1});
                        stat{tep26+2,26} = stat{tep26+2,26}+1;
                    end
                    if isa(alldata{i,3}{j+1},'char')
                        if alldata{i,3}{j+1} == 'L'
                            if isa(alldata{i,3}{j+2},'double')
                                tep26 = find(NB == alldata{i,3}{j-1});
                                stat{tep26+2,26} = stat{tep26+2,26}+1;
                            end
                        end
                    end
                end
            end
        end
    end
end
stat{2,26} = sum(cell2mat(stat(3:NTM+2,26)));

stat{1,27} = '+/- new';
for i = 2:NTM+2
    stat{i,27} = stat{i,5} + stat{i,6} + stat{i,7} + 0.2*stat{i,16} + 0.2*stat{i,23} + 0.2*stat{i,24} + 0.2*stat{i,25} + 0.2*stat{i,26} - stat{i,10} - stat{i,14} - draft{i,10} - 0.5*(draft{i,2}-draft{i,9}-draft{i,10}) - 0.15*stat{i,3};
end

stat{1,29} = '#pass';
stat{1,30} = '#swing';
stat{1,31} = '#dump';
stat{1,32} = '#huck';
stat{1,33} = '#upline';
stat{1,34} = '#break';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'P'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,29} = stat{tep0+2,29}+1;
            end
            if alldata{i,3}{j} == 'S'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,30} = stat{tep0+2,30}+1;
            end
            if alldata{i,3}{j} == 'D'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,31} = stat{tep0+2,31}+1;
            end
            if alldata{i,3}{j} == 'H'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,32} = stat{tep0+2,32}+1;
            end
            if alldata{i,3}{j} == 'U'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,33} = stat{tep0+2,33}+1;
            end
            if alldata{i,3}{j} == 'B'
                tep0 = find(NB == alldata{i,3}{j-1});
                stat{tep0+2,34} = stat{tep0+2,34}+1;
            end
        end
    end
end
for i = 29:34
    stat{2,i} = sum(cell2mat(stat(3:NTM+2,i)));
end

draft{1,11} = 'good pass';
draft{1,12} = 'good swing';
draft{1,13} = 'good dump';
draft{1,14} = 'good huck';
draft{1,15} = 'good upline';
draft{1,16} = 'good break';
for i = 1:Points
    l = length(alldata{i,3});
    for j = 1:l
        if isa(alldata{i,3}{j},'char')
            if alldata{i,3}{j} == 'P'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,11} = draft{tep00+2,11}+1;
                end
            end
            if alldata{i,3}{j} == 'S'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,12} = draft{tep00+2,12}+1;
                end
            end
            if alldata{i,3}{j} == 'D'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,13} = draft{tep00+2,13}+1;
                end
            end
            if alldata{i,3}{j} == 'H'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,14} = draft{tep00+2,14}+1;
                end
            end
            if alldata{i,3}{j} == 'U'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,15} = draft{tep00+2,15}+1;
                end
            end
            if alldata{i,3}{j} == 'B'
                if isa(alldata{i,3}{j+1},'double')
                    tep00 = find(NB == alldata{i,3}{j-1});
                    draft{tep00+2,16} = draft{tep00+2,16}+1;
                end
            end
        end
    end
end
for i = 11:16
    draft{2,i} = sum(cell2mat(draft(3:NTM+2,i)));
end

stat{1,35} = 'pass good%';
stat{1,36} = 'swing good%';
stat{1,37} = 'dump good%';
stat{1,38} = 'huck good%';
stat{1,39} = 'upline good%';
stat{1,40} = 'break good%';
for j = 35:40
    for i = 2:NTM+2
        stat{i,j} = draft{i,j-24} / stat{i,j-6} * 100;
    end
end

xlsx = [Gamen,'.xlsx'];
xlswrite(xlsx,stat)


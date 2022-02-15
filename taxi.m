task = 120;
data = cell(1,task);
coe = zeros(task,4);
fid = fopen('vrp_task.txt');
j = 0;
for i=1:task
    tline = fgetl(fid);
    if isempty(str2num(tline)) == 1
        j = j+1;
        temp = strsplit(tline,{'=',' '});
        coe(j,1) = str2double(temp(2)); %task
        coe(j,2) = str2double(temp(4)); %orders
        coe(j,3) = str2double(temp(6)); %vehicles
        coe(j,4) = str2double(temp(8)); %scores
        taski = cell(coe(j,4),3);
        for k=1:coe(j,4)
            orderk = zeros(1,coe(j,2));
            vehiclesk = zeros(1,coe(j,3));
            tline = fgetl(fid);
            temp = str2num(tline);
            if length(temp) == 4
                orderk(int32(temp(2))) = 1;
                vehiclesk(int32(temp(3))) = 1;
                taski{k,1} = orderk';
                taski{k,2} = vehiclesk';
                taski{k,3} = temp(4);  %costk
            end
            if length(temp) == 5
                orderk(int32(temp(2))) = 1;
                orderk(int32(temp(3))) = 1;
                vehiclesk(int32(temp(4))) = 1;
                taski{k,1} = orderk';
                taski{k,2} = vehiclesk';
                taski{k,3} = temp(5);  %costk
            end
        end
    end
    data{i} = taski;
end
fclose(fid); 

syms a b c d
x=[c];
for i=1:1:100
    if x(length(x))==a
        x=[x,b];
    end
    if x(length(x))==b
        y=rand(1);
        if y<7/17
            x=[x,b];
        else
            x=[x,c];
        end
    end
    if x(length(x))==c
        y=rand(1);
        if y<12/61
            x=[x,b];
        else
            if y<51/61
                x=[x,c];
            else
                x=[x,d];
            end
        end
    end
    if x(length(x))==d
        y=rand(1);
        if y<1/12
            x=[x,a];
        else
            if y<2/3
                x=[x,b];
            else
                if y<5/6
                    x=[x,c];
                else
                    x=[x,d];
                end
            end
        end
    end
end



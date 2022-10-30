function argout = findBlinkFromOpeningClosing(lableSeqCell,tolerance,label)
    arguments
        lableSeqCell,
        tolerance = 10,
        label = {'opening','closing'}
    end

    argout = {};
    if tolerance <=1
        for n = 1:size(lableSeqCell,1)
            location = [];
            for i = 1:size(lableSeqCell,2)-1
                if lableSeqCell(n,i) == label{1} && lableSeqCell(n,i+1) == label{2}
                    location = [location,i];
                end
            end
            argout{n} = location;
        end
    else
        for n = 1:size(lableSeqCell,1)
            location = [];
            for i = 1:size(lableSeqCell,2)-1
                if lableSeqCell(n,i) == label{1} && lableSeqCell(n,i+1) ~= label{1} && i + tolerance < size(lableSeqCell,2)
                    for j = i+1:i+tolerance
                        if lableSeqCell(n,j) == label{2}
                            location = [location,floor(mean([i,j]))];
                            break
                        end
                    end                    
                end
            end
            argout{n} = location;
        end
    end

end
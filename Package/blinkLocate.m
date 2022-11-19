function [blinkLocations] = blinkLocate(signals,mask)
%BLINKLOCATE Summary of this function goes here
%   Detailed explanation goes here
    if isa(signals,'table')
        signals = signals.Variables;
    end

    blinkLocations = [];
    if(size(signals, 1) < size(signals, 2))
        signals = signals';
    end
    if(size(mask, 1) < size(mask, 2))
        mask = mask';
    end
    difSignal = diff(signals);
    mask = mask=='blink';
    index = 1;
    indexRange = size(mask,1)-1;
    while (index<indexRange)
%         %% one direction 
%         while (mask(index) && index<indexRange) 
%             if (difSignal(index)>=0 && difSignal(index+1)<=0)
%                 blinkLocations(end+1) = index;
%             end
%             index = index+1;
%         end
%         index = index+1;
%         

        %% start from center
        left = index; right = indexRange;
        for i = index:indexRange
            
            if (i>1&&~mask(i-1))
                left = i;
            elseif (~mask(i+1))
                right = i;                
                break
            end   
        end
        center = floor((left+right)/2);
%         disp([left,right]);
        if (left~=right)
            step = 1; direction = 1; counter=1; index = center;
            while (index>left && index<right)
                if (direction && difSignal(index)>=0 &&difSignal(index+1)<=0)
                    blinkLocations(end+1) = index+1;
                    index = right;
                    break
                elseif (difSignal(index)<=0 &&difSignal(index-1)>=0)
                    blinkLocations(end+1) = index;
                    index = right;
                    break
                end
                step = floor(counter/2);
                direction = (-1)^counter;
                counter = counter+1;
                index = center + step*direction;            
            end
            index = right+1;
        end
        index = index +1;
    end
end


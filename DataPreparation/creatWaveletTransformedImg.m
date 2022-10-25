function [] = creatWaveletTransformedImg(chopedData, path,slice,channel,sf)
    arguments
        chopedData,
        path,
        slice = 512,
        channel=3,
        sf = 512/4
    end
    originPath = path;
    path = fullfile(path,'WTImg');
    if exist(path)==0
        mkdir(path);
    end

    NameLength = length(num2str(numel(chopedData)));
    for i = 1:numel(chopedData)
        signalLength = slice;
        channelData = [];
        for j = 1:channel    
            data = chopedData{i}(j,:);        
            fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave', 48, ...
                'SamplingFrequency', sf,'FrequencyLimits', [0.3333 40]);        
            [cfs, frq] = fb.wt(data);
            acfs = abs(cfs);
%             acfs = acfs - min(min(acfs));
%             th = quantile(acfs(:) , 0.975); %0.975            
%             acfs(acfs(:) > th) = th;
%             acfs = acfs/th;
            


            if isempty(channelData)
                %channelData = im2uint8(acfs);
                channelData = uint8(acfs);
            else
                %channelData = cat(3,channelData,im2uint8(acfs));
                channelData = cat(3,channelData, uint8(acfs));
            end
        end

        firstName = num2str(i);

        while length(firstName)<NameLength
            firstName = ['0' firstName];
        end

        imFileName = strcat(firstName,'_wt','.jpg');
        imwrite(channelData,fullfile(path,imFileName));
    end


    %disp('wt done');
end
function varargout = detectBlinks(signals,samplingRate,modelName,options)
    arguments
        signals
        samplingRate
        modelName = 'TCNOC'
        options.shiftFactor = 2
        options.tolerance = 10
        options.plot = 0
        options.output = 'location'
    end
    tic;
    shiftFactor = pow2(options.shiftFactor);
    % TODO: auto search on tolerance
    tolerance = options.tolerance;
    plot = options.plot;
    %% reshape
    if (size(signals, 1) < size(signals, 2))
        signals = signals';
    end

    if size(signals, 2) == 1
        signals = [signals,signals,signals/2];
    elseif size(signals, 2) == 2
        signals = [signals(:,1),signals(:,2),mean(signals,2)/2];
    end
    %% filter & resample
    signalLength = size(signals, 1);
    windowLength = 4;
    slice = 512;
    if ~istable(signals)
        signals = array2table(signals);
    end
    filteredSignals = passbandBlinks(signals, samplingRate);
    filteredSignals = array2table(resample(filteredSignals.Variables, slice, windowLength*samplingRate),  'VariableNames', filteredSignals.Properties.VariableNames);
    

    %% apply model

    model = load(['Models\trainedModels\' modelName 'net.mat']).net;    

    Y = {};
    for i = (1:shiftFactor)-1
        X = {};
        
        for j = 1+(slice/shiftFactor)*i:slice:size(signals,1)-slice
            X{end+1,1} = table2array(signals(j:j+slice-1,:))';        
        end
    
        if modelName == "WTCNN"
            segY = cell(numel(X),1);
            fb = cwtfilterbank('SignalLength',slice,'VoicesPerOctave', 48, ...
                'SamplingFrequency', samplingRate,'FrequencyLimits', [0.3333 20]);  
            for j = 1:numel(X)
                img = [];
                [cfs, ~] = fb.wt(X{j});
                acfs = abs(cfs);
                acfs = acfs - min(min(acfs));
                th = quantile(acfs(:) , 0.975); %0.975            
                acfs(acfs(:) > th) = th;
                acfs = acfs/th;
                if isempty(img)
                    img = im2uint8(acfs);
                else
                    img = cat(3,img,im2uint8(acfs));
                end
                [segY{j},~,~]=semanticseg(img,model);
            end
            Y{end+1} = segY;
        else 
            Y{end+1} = classify(model, X, "MiniBatchSize",1);

        end

        lable = [];        
        for k = 1:size(Y{i+1})
            lable = [lable,Y{i+1}{k}];
        end 
        Y{i+1} = lable;
    end
    
    %% summarize shifted labels
    % beginBank = cell(shiftFactor-1); endBank = cell(shiftFactor-1);
    
    shiftedY = categorical(zeros(shiftFactor,signalLength));
    for i = 1:shiftFactor
        shift = (slice/shiftFactor)*(i-1);
        shiftedY(i ,1+shift:size(Y{i},2)+shift) = Y{i};
    end
    
    blinkMasks = weightingLable(shiftedY);
    while numel(blinkMasks) < signalLength
        blinkMasks = [blinkMasks, 'n/a'];
    end

    blinkLocations = findBlinkFromOpeningClosing(blinkMasks,tolerance);
    blinkLocations = blinkLocations{1};
    blinkLocations = floor(blinkLocations/slice*(windowLength*samplingRate));

    %% plot
    if plot
        coordinate = signals.Var1(blinkLocations);

        figure();

        hold on;
        plot(signals.Var1');
        plot(location,coordinate,'r*');
        hold off;
        %plot();
    end


    %% output
    output = options.output;
    nargout = numel(output); % sum(ismember(output,','))+1;
    for k = 1:nargout
        if contains(output{k},'location')
            varargout{k} = blinkLocations;
%             output = output(~ismember(output,'location'));
        elseif contains(output{k},'mask')
            varargout{k} = blinkMasks;
%             output = output(~ismember(output,'mask'));
        elseif contains(output{k},'time')
            varargout{k} = toc;
%             output = output(~ismember(output,'time'));
        end
    end 
end

 
%% helper function
function weightedSeq = weightingLable(seqArray)
    length = max(size(seqArray));
    weightedSeq = [];
    for i = 1:length
        thisTime = seqArray(:,i);
        thisTime = thisTime(~ismember(thisTime,['0','n/a']));
        if ~isempty(thisTime)
            weightedSeq = [weightedSeq, mode(thisTime)];
        else
            weightedSeq = [weightedSeq, 'n/a'];
        end
    end
end

function img = WTImg(chopedData, slice,channel,sf)

end

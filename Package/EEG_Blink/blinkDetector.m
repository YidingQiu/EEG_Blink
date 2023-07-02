classdef blinkDetector
    properties
        signals
        samplingRate
        modelName
        options detectOptions % options class
    end
    
    methods
        function obj = blinkDetector(signals, samplingRate, modelName, options)
            arguments
                signals = table
                samplingRate = 1
                modelName = 'TCNOC'
                options = detectOptions()
            end
            obj.signals = signals;
            obj.samplingRate = samplingRate;
            obj.modelName = modelName;
            obj.options = options;
        end
        
        function varargout = detectBlinks(obj,progressBar)
            if sum(contains(obj.options.output,'time')) > 0
                tic; 
            end
            shiftFactor = pow2(obj.options.shiftFactor);
            tolerance = obj.options.tolerance;
            plotOption = obj.options.plot;
            
            %% reshape
            if (size(obj.signals, 1) < size(obj.signals, 2))
                obj.signals = obj.signals';
            end

            if size(obj.signals, 2) == 1
                obj.signals = [obj.signals,obj.signals,obj.signals/2];
            elseif size(obj.signals, 2) == 2
                obj.signals = [obj.signals(:,1),obj.signals(:,2),mean(obj.signals,2)/2];
            end
                        
            %% filter & resample
            signalLength = size(obj.signals, 1);
            windowLength = 4;
            slice = 512;
            if ~istable(obj.signals)
                obj.signals = array2table(obj.signals);
            end
            filteredSignals = passbandBlinks(obj.signals, obj.samplingRate);
            filteredSignals = array2table(resample(filteredSignals.Variables, slice, windowLength*obj.samplingRate),  'VariableNames', filteredSignals.Properties.VariableNames);
        %     Ds=load('Ds.mat').Ds;
            % filteredSignals = Ds{1}{1};
            %% apply model
        
            model = load(['Models\trainedModels\' obj.modelName 'net.mat']).net;    
        
            Y = {};

            for i = (1:shiftFactor)-1
                X = {};
                if obj.options.shortSignal % for runtime check 
                    X = table2array(filteredSignals);
                else
                    for j = 1+(slice/shiftFactor)*i:slice:size(filteredSignals,1)-slice
                        X{end+1,1} = table2array(filteredSignals(j:j+slice-1,:))';        
                    end
                end
            
                if obj.modelName == "WTCNN"
                    segY = cell(numel(X),1);
                    fb = cwtfilterbank('SignalLength',slice,'VoicesPerOctave', 48, ...
                        'SamplingFrequency', obj.samplingRate,'FrequencyLimits', [0.3333 20]);  
                    for j = 1:numel(X)
                        img = [];
                        for k = 1:3
                            [cfs, ~] = fb.wt(X{j}(k,:));
                            acfs = abs(cfs);
                            acfs = acfs - min(min(acfs));
            %                 th = quantile(acfs(:) , 0.975); %0.975            
            %                 acfs(acfs(:) > th) = th;
            %                 acfs = acfs/th;
                        
                            if isempty(img)
            
            %                     img = im2uint8(acfs);
                                img = uint8(acfs);
                            else
            %                     img = cat(3,img,im2uint8(acfs));
                                img = cat(3,img,uint8(acfs));
                            end
                        end
                        img = ones([270 512 3]);
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

                waitbar(i / shiftFactor, progressBar);
                drawnow;

            end
            
            %% summarize shifted labels
            % beginBank = cell(shiftFactor-1); endBank = cell(shiftFactor-1);
            
            shiftedY = categorical(zeros(shiftFactor,signalLength));
            for i = 1:shiftFactor
                shift = (slice/shiftFactor)*(i-1);
                shiftedY(i ,1+shift:size(Y{i},2)+shift) = Y{i};
            end
            
            blinkMasks = obj.weightingLable(shiftedY);
            while numel(blinkMasks) < signalLength
                blinkMasks = [blinkMasks, 'n/a'];
            end
            
            if sum(strcmp(categories(blinkMasks),"opening")) % oc 
                blinkLocations = findBlinkFromOpeningClosing(blinkMasks,tolerance);
                blinkLocations = blinkLocations{1};        
            else % blink or not
                blinkLocations = blinkLocate(filteredSignals,blinkMasks);
            end
            blinkLocations = floor(blinkLocations/slice*(windowLength*obj.samplingRate)); % 
            masks = blinkMasks;
        %     for k = 1:signalLength
        %         index = floor(k/(windowLength*samplingRate)*slice)+1;
        %         masks(k)=
        %     blinkMasks = resample(blinkMasks, windowLength*samplingRate, slice);
            %% plot
            if plotOption
                plotSignal = obj.signals.Fp1;        
                coordinate = plotSignal(blinkLocations,1);
                figure();
                hold('on');
                plot(1:size(plotSignal,1),plotSignal');
                plot(blinkLocations,coordinate,'r*');
                hold('off');
                %plot();
            end
        
        %% output
        output = obj.options.output;
        nargout = numel(output);
        for k = 1:nargout
            if contains(output{k},'location')
                varargout{k} = blinkLocations;
            elseif contains(output{k},'mask')
                varargout{k} = masks;
            elseif contains(output{k},'time')
                varargout{k} = toc;
            end
        end 
        %close(h);
    end
        
        %% helper function
        function weightedSeq = weightingLable(obj, seqArray)
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
    end
end

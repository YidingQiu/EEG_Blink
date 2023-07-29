function lr = generateResettingLearningRate(numIterations,...
                learningRateRange,...
                numPeaks)
            arguments
                numIterations       (1,1) {mustBeInteger, mustBePositive}
                learningRateRange   (1,2) {mustBePositive}
                numPeaks            (1,1) {mustBeInteger, mustBePositive}
            end
            if(numPeaks == 0)
                numPeaks = 1;
            end
           
            x = 1:numIterations;
            % rate of decay
            lambda = log(learningRateRange(2)/learningRateRange(1))/numIterations;    
            % exponential decay
            y1 = learningRateRange(1)*exp(lambda*x);            
            % generate a saw like signal
            y2 = 2.1*(sawtooth((numPeaks)/numIterations*2*pi*x, 0) + 1)/2 + 1;
            y2(end) = 1;
            y2(1:floor(numIterations/(numPeaks))) = 1;
            % modulate
            lr = (y1 + y1 .* y2)/2;            
        end
        %-----------------------------------------------------------------

function bs = generateResettingBatchSize(numIterations, batchSizeRange, numPeaks)
    arguments
        numIterations   (1,1) {mustBeInteger, mustBePositive}
        batchSizeRange  (1,2) {mustBeInteger, mustBePositive}
        numPeaks        (1,1) {mustBeInteger, mustBePositive}
    end

    if(numPeaks == 0)
        numPeaks = 1;
    end

    if(batchSizeRange(1) == batchSizeRange(2))
        bs = batchSizeRange(1) * ones(1, numIterations);
    else
        x = 1:numIterations;
        % rate of decay
        lambda = log(batchSizeRange(2)/(2*batchSizeRange(1)))/numIterations;    
        % exponential decay
        y1 = (2*batchSizeRange(1))*exp(lambda*x);            
        % generate a saw like signal
        y2 = (sawtooth(numPeaks/numIterations*2*pi*x, 0) + 1)/2 + 1;
        y2(end) = 1;
       
        bs = ceil(y1./y2);
    end
end
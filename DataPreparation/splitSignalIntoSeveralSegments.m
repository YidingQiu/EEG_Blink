.function [searchTrain,testSave,validationTest] = splitSignalIntoSeveralSegments(signal, datasetSplitRatio, step, segLength)

    if(size(signal, 1) < size(signal, 2))
        signal = signal';
    end
    numSegments = floor(size(signal, 1)/(1.5*segLength));
%     segmentSet = reshape(signal(1:numSegments*segLength), segLength, numSegments);

    searchTrainRatio = datasetSplitRatio(1); 
    testSaveRatio = datasetSplitRatio(2);
    validationTestRatio = datasetSplitRatio(3);
    
    % for loop here if need to split more
    searchTrainIndex = sort(randperm(numSegments, round(searchTrainRatio*numSegments)));
    leftIndex=setdiff(1:numSegments, searchTrainIndex);
    testSaveIndexIndex = sort(randperm(numel(leftIndex), round(testSaveRatio/(1-searchTrainRatio)*numel(leftIndex))));
    testSaveIndex = leftIndex(testSaveIndexIndex);
    validationTestIndex = setdiff(1:numSegments, [searchTrainIndex testSaveIndex]);

    [setOfSegments1, searchTrain] = splitSignalIntoSegmentsWithIndexes(signal, searchTrainIndex, step, segLength);
    [setOfSegments2, testSave] = splitSignalIntoSegmentsWithIndexes(signal, testSaveIndex, step, segLength);
    [setOfSegments3, validationTest] = splitSignalIntoSegmentsWithIndexes(signal, validationTestIndex, step, segLength);



    
    
%     trainIndex = sort(randperm(numSegments, round(trainingDatasetRatio*numSegments)));
%     testIndex = setdiff(1:numSegments, trainIndex);
% 
%     [setOfSegmentsTrain, indexesTrain] = splitSignalIntoSegmentsWithIndexes(signal, trainIndex, step, segLength);
%     [setOfSegmentsTest,  indexesTest] = splitSignalIntoSegmentsWithIndexes(signal, testIndex, step, segLength);
end

function [setOfSegments, indexOfSet] = splitSignalIntoSegmentsWithIndexes(signal, indexes, step, segLength)
    setOfSegments = [];
    indexOfSet = [];
    k = 1;
    while(k <= length(indexes))
%         k
        l = 0;
        while(k + l + 1 < length(indexes) && (indexes(k + l + 1) - indexes(k+l)) == 1)
          l = l + 1;
        end
    
%       tempSegment     = signal((indexes(k)-1) * 2*segLength+1:(indexes(k) + l) * 2*segLength, :);
        segmentsTemp    = zeros((indexes(k+l)-indexes(k) + 1) * round(1.5*segLength/step) -  segLength/step, size(signal,2), segLength);
        indexesTemp     = [];
        offset = (indexes(k)-1) * round(1.5*segLength);
        for m = 0:(indexes(k+l)-indexes(k) + 1) * round(1.5*segLength/step) -  segLength/step %(length(tempSegment)/step - segLength/step)
            indexesTemp(m + 1, :) = offset + ((m*step+1):m*step+segLength);
            if(istable(signal))
                segmentsTemp(m + 1, :, :) = signal(indexesTemp(m + 1, :), :).Variables';
            else
                segmentsTemp(m + 1, :, :) = signal(indexesTemp(m + 1, :), :);
            end
        end
        setOfSegments   = [setOfSegments; segmentsTemp];
        indexOfSet         = [indexOfSet; indexesTemp];
        k = k + l + 1;
    end
end
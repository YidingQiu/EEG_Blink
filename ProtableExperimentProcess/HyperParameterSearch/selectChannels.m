function newXTrain = selectChannels(XTrain, i)
    % Initialize the new cell array
    newXTrain = cell(size(XTrain));

    % Iterate over each element in XTrain
    for k = 1:numel(XTrain)
        % Select only the channels specified by i
        newXTrain{k} = XTrain{k}(i, :);
    end
end

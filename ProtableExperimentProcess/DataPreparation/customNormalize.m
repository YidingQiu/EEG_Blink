function normalizedData = customNormalize(dataTable)
    % Initialize an empty table for the normalized data
    normalizedData = dataTable;
    
    % Loop over each column in the table
    for i = 1:width(dataTable)
        % Get the data for the current column
        columnData = dataTable{:, i};
        
        % Calculate the 10th and 90th percentiles
        lowerBound = prctile(columnData, 10);
        upperBound = prctile(columnData, 80);
        
        % Ignore data below the 10th percentile and above the 90th percentile
        highConfidence = columnData;
        highConfidence(columnData < lowerBound|columnData > upperBound) = 0;
        removed = highConfidence;
        removed = removed(~isnan(removed));
        
        % Center the data and scale it to have a standard deviation of 1
        columnData = (columnData - mean(highConfidence)) / std(removed);
        
        % Store the normalized data in the output table
        normalizedData{:, i} = columnData;
    end
end

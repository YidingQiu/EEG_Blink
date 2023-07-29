function normalizedData = normalizeData(dataTable, epsilon)

    arguments
        dataTable
        epsilon = 1e-6
    end
    
    normalizedData = array2table(zeros(size(dataTable)));    
    for i = 1:width(dataTable) 
        mu = mean(dataTable{:, i}); 
        sigma = std(dataTable{:, i});
        
        normalizedData{:, i} = (dataTable{:, i} - mu) / (sigma + epsilon);
    end
end
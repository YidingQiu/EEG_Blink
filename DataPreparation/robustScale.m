function robustScaledData = robustScale(dataTable, q1, q3)
    
    arguments
        dataTable
        q1 = 0.25
        q3 = 0.75
    end

    robustScaledData = array2table(zeros(size(dataTable)));     
    for i = 1:width(dataTable) 
        median_val = median(dataTable{:, i}); 
        iqr_val = iqr(dataTable{:, i});
       
        robustScaledData{:, i} = normalizeData((dataTable{:, i} - median_val) / iqr_val);
    end
end

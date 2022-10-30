function [accuracy,TP,FP,FN] = testOC(YTest,YPred)
    accuracy=0;
    valError = 0;
    if iscell(YPred)
        for k = 1:length(YPred)
            valError = valError + mean(YPred{k} == YTest{k});
        end
        accuracy = valError/length(YPred);
    else 
        
    end
    TP = 0;FP = 0;FN=0;

    for i = 1:numel(YTest)
        gt = findBlinkFromOpeningClosing(YTest{i});
        if iscell(YPred)
            p = findBlinkFromOpeningClosing(YPred{i});
        else
            p = findBlinkFromOpeningClosing(YPred(:,i)');
        end
        gt = gt{1};p=p{1};
        
        TP = TP + sum(ismembertol(gt,p,50));%numel(intersect(gt,p))
        if (numel(gt) > numel(p))
            FN = FN + numel(gt) - numel(p);
        else
            FP = FP + numel(p) - numel(gt);
        end
      
    end 

end
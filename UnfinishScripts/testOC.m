function [TP,FP,FN] = testOC(Ytest,YPred)

    groundTruth = {};
    predicted = {};
    TP = 0;FP = 0;FN=0;

    for i = 1:numel(Ytest)
        gt = findBlinkFromOpeningClosing(YTest{i});
        p = findBlinkFromOpeningClosing(YPred{i});
        gt = gt{1};p=p{1};
        groundTruth{end+1} = gt;
        predicted{end+1} = p;
        
        TP = TP + numel(intersect(gt,p));
        if (numel(gt) > numel(p))
            FN = FN + numel(gt) - numel(p);
        else
            FP = FP + numel(p) - numel(gt);
        end
      
    end 
%     TP,FP,FN,errorList
    Precision=TP/(TP+FP),Recall=TP/(TP+FN)


end
% Function to evaluate the performance of an eye blink detection algorithm
% by comparing ground truth (YTest) with predicted results (YPred).
% Returns accuracy, true positives (TP), false positives (FP), and false negatives (FN).
function [accuracy, TP, FP, FN] = testOC(YTest, YPred)
    % Initialize accuracy and validation error
    accuracy = 0;
    valError = 0;

    % If YPred is a cell array, iterate over each element and compute the mean error
    if iscell(YPred)
        for k = 1:length(YPred)
            valError = valError + mean(YPred{k} == YTest{k});
        end
        % Compute the final accuracy by dividing the total error by the number of elements in YPred
        accuracy = valError / length(YPred);
    else 
        % If YPred is not a cell array, this part can be updated to handle other data types
    end

    % Initialize true positives (TP), false positives (FP), and false negatives (FN)
    TP = 0;
    FP = 0;
    FN = 0;

    % Iterate over each element in YTest
    for i = 1:numel(YTest)
        % Find ground truth eye blinks using the findBlinkFromOpeningClosing function
        gt = findBlinkFromOpeningClosing(YTest{i});

        % Find predicted eye blinks using the findBlinkFromOpeningClosing function
        if iscell(YPred)
            p = findBlinkFromOpeningClosing(YPred{i});
        else
            p = findBlinkFromOpeningClosing(YPred(:, i)');
        end

        % Extract the blink data from the cell arrays
        gt = gt{1};
        p = p{1};

        % Compute true positives with a tolerance of 50
        TP = TP + sum(ismembertol(gt, p, 50));

        % Compute false negatives and false positives
        if (numel(gt) > numel(p))
            FN = FN + numel(gt) - numel(p);
        else
            FP = FP + numel(p) - numel(gt);
        end
    end
end

%%%%%
% function [accuracy,TP,FP,FN] = testOC(YTest,YPred)
%     accuracy=0;
%     valError = 0;
%     if iscell(YPred)
%         for k = 1:length(YPred)
%             valError = valError + mean(YPred{k} == YTest{k});
%         end
%         accuracy = valError/length(YPred);
%     else 
%         
%     end
%     TP = 0;FP = 0;FN=0;
% 
%     for i = 1:numel(YTest)
%         gt = findBlinkFromOpeningClosing(YTest{i});
%         if iscell(YPred)
%             p = findBlinkFromOpeningClosing(YPred{i});
%         else
%             p = findBlinkFromOpeningClosing(YPred(:,i)');
%         end
%         gt = gt{1};p=p{1};
%         
%         TP = TP + sum(ismembertol(gt,p,50));%numel(intersect(gt,p))
%         if (numel(gt) > numel(p))
%             FN = FN + numel(gt) - numel(p);
%         else
%             FP = FP + numel(p) - numel(gt);
%         end
%       
%     end 
% 
% end
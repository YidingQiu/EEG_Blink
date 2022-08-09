classdef ctcClassificationLayer < nnet.layer.ClassificationLayer
    
    
        properties
       
            % Categories (column categorical array) 
            Categories
           
            % cache for subresults to avoid recalculating  subproblems over and over again
            cacheSub
            
        end
    
    methods
        function layer = ctcClassificationLayer(name,categories) 
            % layer = ctcClassificationLayer(name,categories) creates Connectionist Temporal Classification Layer
            % with specifies the layer name and categories.
            layer.Name = name;
            layer.Description = 'Connectionist Temporal Classification Layer';
            
            if numel(categories) == 0
                layer.NumClasses = [];
            else
                layer.NumClasses = numel(categories);
            end   
            
            layer.Categories = categories;
        end
       function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     � Predictions made by network
            %         T     � Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            % Layer forward loss function goes here.
            
              % Calculate sum of squares.
            loss = log(ctcLabelingProb(T, Y));

            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function.returns the derivatives of
            % the CTC  loss with respect to the predictions Y.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     � Predictions made by network
            %         T     � Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y
            % Layer backward loss function goes here.
%             [maxT, maxC]= size(T);
%             
%         	labelIdx = 0;      	   
%             lastMaxIdx = maxC;
%             for t=1:1:maxT 
%                 maxIdx = max(T(t, :));
%                 if ((maxIdx ~= lastMaxIdx) && (maxIdx ~=  layer.NumClasses))
%                     labelIdx = labelIdx +  layer.Categories(maxIdx);
%                 end
%                 lastMaxIdx = maxIdx;
%             end
%             
%             dLdY = labelIdx;
            dLdY = log(ctcLabelingProb(Y, T));

            
        end

        function ctcProb = ctcLabelingProb(T, Y)
            % size of input matrix
            [maxT, ~] = size(T);
            
            labelGroundTruth = GroundTruth(Y, layer.Categories, layer.NumClasses);
            
            % pre-allocate cache
            layer.cacheSub = nan(maxT, numel(labelingWithBlanks));
            
            ctcProb =  recLabelingProb(maxT-1, numel(labelGroundTruth)-1, T, labelGroundTruth) + recLabelingProb(maxT-1, numel(labelGroundTruth)-2, T, labelGroundTruth); 
            
        end
        
        function lgt = GroundTruth (mY, mCategories, mNum)
            for i=0:size(mY,1)
                lgt = vertcat(mY(i), mCategories(mNum));
            end
            
        end
        
        function recLabelingVal = recLabelingProb(t, s, T, mGroundTruth)
        % recursively compute probability of labeling, save results of sub-problems in cache to avoid recalculating them
        
         % check index of labeling 
         	if (s < 0) recLabelingVal = 0; end
          % check  sub-problem already computed 
        	if (layer.cacheSub(t,s) ~= Nan)
                recLabelingVal = layer.cacheSub(t,s);
            end
            
            % Calc initial value
         	if t == 0 
                if s == 0 
                    recLabelingVal = T(0, layer.NumClasses);
                elseif s == 1 
                    recLabelingVal = T(0, mGroundTruth(1)); 
                else 
                    recLabelingVal = 0;
                end
                layer.cacheSub(t,s) = recLabelingVal; 
            end
            
            %  recursion on s and t 
            recLabelingVal = (recLabelingProb(t-1, s, T, mGroundTruth) + recLabelingProb(t-1, s-1, T, mGroundTruth)) * T(t, mGroundTruth(s)); 
            % in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done 
            if (mGroundTruth(s) == NumClasses)  || ((s >= 2) && (mGroundTruth(s-2) == mGroundTruth(s))) 
                layer.cacheSub(t,s) = recLabelingVal;
            end
            
            % otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1 
            recLabelingVal = recLabelingVal+ recLabelingProb(t-1, s-2, T, mGroundTruth) * T(t, mGroundTruth(s)); 
            layer.cacheSub(t,s) = recLabelingVal;              
 
        end
                
     end
    
    methods(Access=private)
    
    end
end
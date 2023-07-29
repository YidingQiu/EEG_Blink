classdef mCrossEntropy < nnet.internal.cnn.layer.ClassificationLayer
    % CrossEntropy   Cross entropy loss output layer
    
    %   Copyright 2015-2022 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % InputNames   Crossentropy layer has one input
        InputNames = {'in'}
        
        % OutputNames   Crossentropy layer has no outputs
        OutputNames = {}
    end
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % Categories (column categorical array) The categories of the classes
        % It can store ordinality of the classes as well.
        Categories
        
        % ClassWeights   A vector of weights. Either [] or the same size as
        % number of class names.
        ClassWeights
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'classoutput'
    end

    properties (SetAccess = private)
        % NumClasses (scalar int)   Number of classes
        NumClasses
        
        % ObservationDim (scalar int)   The dimension of the input data
        % along which holds the number of observations within the data.
        ObservationDim
        
        % HasSizeDetermined   True for layers with size determined. This
        % value should always be false so that the layer can be correctly
        % configured to operate in other network architectures with
        % different observation dimensions.
        HasSizeDetermined = false;
        
        % InternalClassWeights   A vector of weights equal to class weights
        % or ones, with the same dimensions as the input data.
        InternalClassWeights
    end
    
    methods
        function this = mCrossEntropy(name, numClasses, categories, ...
                classWeights, observationDim)
            % Output  Constructor for the layer
            % creates an output layer with the following parameters:
            %
            %   name                - Name for the layer
            %   numClasses          - Number of classes. [] if it has to be
            %                       determined later
            %   categories          - Categories of the output layer
            %   classWeights        - The weights for each class in
            %                       categories
            %   observationDim      - The dimension of the input data
            %                         along which holds the number of 
            %                         observations within the data
            this.Name = name;
            this.NumClasses = numClasses;
            this.Categories = categories;
            this.ObservationDim = observationDim;
            this.ClassWeights = classWeights;
            
            % CrossEntropy layer needs Z but not X for the backward pass
            this.NeedsXForBackward = false;
            this.NeedsZForBackward = true;
        end
        
        function this = configureForInputs(this, Xs)
            % A cross-entropy layer doesn't have any learnable parameters
            % and should be reusable in new networks. Therefore,
            % configureForInputs always recomputes the number of dimensions 
            % in the input size and stores it in the layer (even if it was
            % already configured).
            X = Xs{1};
            this.assertValidInputSize(X);
            
            % Configure the number of classes
            if isempty(this.NumClasses)
                %this.NumClasses = getSizeForDims(X,'C');
                this.NumClasses = 3;
            end
            
            % Configure the observation dimension
            obsDim = finddim(X,'B');
            if isempty(obsDim)
                % If X does not contain an observation dimension, we
                % assume that the data layout is Sx..xSxCxB
                obsDim = numel(getSizeForDims(X,'SC'))+1;
            end            
            this.ObservationDim = obsDim;
            this.InternalClassWeights = iGetInternalWeights(...
                this.ClassWeights, this.ObservationDim, this.NumClasses);
        end
        
        function Xs = forwardExampleInputs(this, Xs)
            this.assertValidInputSize(Xs{1})
        end
        
        function outputSeqLen = forwardPropagateSequenceLength(~, ~, ~)
            outputSeqLen = {};
            error("Temporary internal error: forwardPropagateSequenceLength "+...
                "should not be called on a CrossEntropy layer anymore")
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            % learnable parameters
        end
        
        function this = set.Categories( this, val )
            if isequal(val, 'default')
                this.Categories = iDefaultCategories(this.NumClasses); %#ok<MCSUP>
            elseif iscategorical(val)
                % Set Categories as a column array.
                if isrow(val)
                    val = val';
                end
                this.Categories = val;
            else
                warning('Invalid value in set.Categories.');
            end
        end
        
        function this = set.ClassWeights( this, val )
            this.ClassWeights = val;
            this.InternalClassWeights = iGetInternalWeights(...
                this.ClassWeights, this.ObservationDim, this.NumClasses); %#ok<MCSUP>
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
        end
        
        function this = setupForGPUPrediction(this)
        end
        
        function this = setupForHostTraining(this)
        end
        
        function this = setupForGPUTraining(this)
        end
        
        function loss = forwardLoss( this, Y, T )
            % forwardLoss    Return the cross entropy loss between estimate
            % and true responses averaged by the number of observations
            %
            % Syntax:
            %   loss = layer.forwardLoss( Y, T );
            %
            % Vector Inputs:
            %   Y   Predictions made by network, numClasses-by-numObs-by-seqLength
            %   T   Targets (actual values),  numClasses-by-numObs-by-seqLength
            %
            % Image Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs
            %
            % Volume Inputs:
            %   Y   Predictions made by network, 1-by-1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-1-by-numClasses-by-numObs
            %
            % N-D Inputs:
            %   Y   Predictions made by network, 1-by-1-by- ...(extended to N dimensions)-by-numClasses-by-numObs where N is input size dimension
            %   T   Targets (actual values), 1-by-1-by- ...(extended to N dimensions)-by-numClasses-by-numObs  where N is input size dimension
            
            % Validate response and target size.
            Y = sscb2cbt(Y);
            if ~isequal(size(Y), size(T))
                iErrorIfResponsesAndTargetsAreMismatched(size(Y), size(T))
            end

            % Take the product of numObs and seqLength.
            numElems = prod( size(Y, [this.ObservationDim this.ObservationDim+1]) );
            % In the default case, class weights is a vector of ones.
            wT = this.applyClassWeights(T);
            loss = -sum(  (wT.*log(nnet.internal.cnn.util.boundAwayFromZero(Y)))./numElems, 'all' );
        end
        
        function dX = backwardLoss( this, Y, T )
            % backwardLoss    Back propagate the derivative of the loss
            % function
            %
            % Syntax:
            %   dX = layer.backwardLoss( Y, T );
            %
            % Vector Inputs:
            %   Y   Predictions made by network,  numClasses-by-numObs-by-seqLength
            %   T   Targets (actual values),  numClasses-by-numObs-by-seqLength
            %
            % Image Inputs:
            %   Y   Predictions made by network, 1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-numClasses-by-numObs
            %
            % Volume Inputs:
            %   Y   Predictions made by network, 1-by-1-by-1-by-numClasses-by-numObs
            %   T   Targets (actual values), 1-by-1-by-1-by-numClasses-by-numObs
            %
            % N-D Inputs:
            %   Y   Predictions made by network, 1-by-1-by- ...(extended to N dimensions)-by-numClasses-by-numObs where N is input size dimension
            %   T   Targets (actual values), 1-by-1-by- ...(extended to N dimensions)-by-numClasses-by-numObs  where N is input size dimension
            
            % Validate response and target size.
            Y = sscb2cbt(Y);
            if ~isequal(size(Y), size(T))
                iErrorIfResponsesAndTargetsAreMismatched(size(Y), size(T))
            end

            % In the default case, class weights is a vector of ones.
            wT = this.applyClassWeights(T);
            numObservations = size(Y, this.ObservationDim);
            dX = (-wT./nnet.internal.cnn.util.boundAwayFromZero(Y))./numObservations;
        end
    end
    
    methods (Access = private)
        function assertValidInputSize(this, ~)
            % The input size X must be consistent with the number of
            % classes the layer has been set on construction
            
%             inputSize = getSizeForDims(X, 'SC');

            inputSize = [512 3];
            
            if isempty(this.NumClasses)
                if ~(iHasValidSpatialDims(inputSize) || isscalar(inputSize))
                    error(message('nnet_cnn:layer:ClassificationLayer:MustHaveSingletonSpatialInputs'));
                end
            else
                numClasses = inputSize(end);
                if ~isequal(numClasses, this.NumClasses)
                    error(message('nnet_cnn:layer:ClassificationLayer:InvalidInputData',...
                        inputSize(end), this.NumClasses));
                end
                
                if ~isequal(inputSize, [ones(1,numel(inputSize)-1) this.NumClasses])
                    error(message('nnet_cnn:layer:ClassificationLayer:MustHaveSingletonSpatialInputs'));
                end
            end
        end
        
        function wT = applyClassWeights(this, T)
            % Assign weight, W(c) to each observation that belongs to
            % class(c).           
            wT = T .* this.InternalClassWeights;
        end
    end
end

function tf = iHasValidSpatialDims(inputSize)
if numel(inputSize) >= 2
    tf = isequal(inputSize(1:numel(inputSize)-1), ones(1,numel(inputSize)-1));
else
    tf = false;
end
end

function cats = iDefaultCategories(numClasses)
% Set the default Categories
cats = categorical(1:numClasses)';
end

function internalWeights = iGetInternalWeights(weights, obsDim, numClasses)
% Get internal class weights for the layer. By default, internal class
% weights is ones.
if isempty(weights)
    weights = ones( [512 numClasses 1] );
end
internalWeights = shiftdim(shiftdim(weights, 2),-1);
end

function iErrorIfResponsesAndTargetsAreMismatched(szY, szT)
sizeY = iSizeToString(szY);
sizeT = iSizeToString(szT);
error( message('nnet_cnn:internal:cnn:layer:OutputLayer:NetworkPredictionAndTargetMismatch', sizeY, sizeT) );
end

function str = iSizeToString(sz)
str = join(string(sz), matlab.internal.display.getDimensionSpecifier);
end

function timeSeries = sscb2cbt(A)
A = extractdata(A);
timeSeries=dlarray(single(reshape(permute(A,[1 4 2 3]),[3,1,512])),"CBT");
end
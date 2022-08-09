classdef mClassificationOutputLayer < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % ClassificationOutputLayer   Classification output layer
    %
    %   To create a classification output layer, use classificationLayer
    %
    %   A classification output layer. This layer is used as the output for
    %   a network that performs classification.
    %
    %   ClassificationOutputLayer properties:
    %       Name             - Name for the layer
    %       Classes          - Categories into which the input data is 
    %                          classified
    %       ClassWeights     - Weight assigned to each class
    %       OutputSize       - Size of the output
    %       LossFunction     - Loss function that is used for training
    %       NumInputs        - Number of inputs for the layer
    %       InputNames       - Names of the inputs of the layer
    %
    %   Example:
    %       Create a classification output layer.
    %
    %       layer = classificationLayer();
    %
    %   See also classificationLayer
    
    %   Copyright 2015-2020 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name
        
        % Classes (categorical)  The categories into which the input data
        % is classified.
        %   A categorical column vector whose elements are the distinct
        %   classes to classify the input data to the network. It can be
        %   set passing a string or categorical vector, or a cell vector of
        %   character vectors, or 'auto'. When 'auto' is specified, the
        %   classes are automatically set during training. Default: 'auto'.
        Classes
        
        % ClassWeights   Weights for weighted cross entropy loss
        %   A vector of nonnegative weights or 'none'. Each element
        %   specifies the weight of the corresponding class in Classes.
        %   Default: 'none'.
        ClassWeights
    end

    properties(SetAccess = private, Dependent, Hidden)
        % ClassNames   The names of the classes
        %   A cell array containing the names of the classes.
        ClassNames
    end
    
    properties(SetAccess = private, Dependent)       
        % OutputSize   The size of the output
        %   The size of the output. This will be determined at training
        %   time. Prior to training, it is set to 'auto'.
        OutputSize
    end
    
    properties(SetAccess = private)
        % LossFunction   The loss function for training
        %   The loss function that will be used during training. Possible
        %   values are:
        %       'crossentropyex'    - Cross-entropy for exclusive outputs.
        LossFunction = 'crossentropyex';
    end
    
    methods
        function this = mClassificationOutputLayer(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 4.0;
            out.Name = privateLayer.Name;
            out.ObservationDim = privateLayer.ObservationDim;
            out.NumClasses = privateLayer.NumClasses;            
            out.Categories = privateLayer.Categories;
            out.ClassWeights = privateLayer.ClassWeights;
        end
        
        function val = get.OutputSize(this)
            if(isempty(this.PrivateLayer.NumClasses))
                val = 'auto';
            else
                val = this.PrivateLayer.NumClasses;
            end
        end
        
        function val = get.ClassNames(this)
            val = this.PrivateLayer.ClassNames(:);
        end
        
        function val = get.Classes(this)
            if isempty(this.PrivateLayer.Categories)
                val = 'auto';
            else
                val = this.PrivateLayer.Categories(:);
            end
        end
        
        function this = set.Classes(this, val) 
            iAssertValidClasses(val);
            classes = iConvertClassesToCanonicalForm(val);
            try
                iCheckConsistencyClassesAndClassWeights(...
                    classes, this.PrivateLayer.ClassWeights);
            catch exception
                iAddSolutionAndRethrow(exception);
            end
            this = this.updateClasses(classes);
        end
        
        function val = get.ClassWeights(this)
            if isempty(this.PrivateLayer.ClassWeights)
                val = 'none';
            else
                val = this.PrivateLayer.ClassWeights;
            end
        end
        
        function this = set.ClassWeights(this, val)
            val = gather(val);
            iAssertValidClassWeights(val);
            classWeights = iConvertClassWeightsToCanonicalForm(val);
            iCheckConsistencyClassesAndClassWeights(...
                this.PrivateLayer.Categories, classWeights);
            this.PrivateLayer.ClassWeights = classWeights;
        end
        
        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end
    end
    
    methods(Hidden, Static)
        function inputArguments = parseInputArguments(varargin)
            varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
            parser = iCreateParser();
            parser.parse(varargin{:});
            inputArguments = iConvertToCanonicalForm(parser);
            iCheckConsistencyClassesAndClassWeights(...
                inputArguments.Classes, inputArguments.ClassWeights);
        end
        
        function this = loadobj(in)
            if in.Version <= 1
                in = iUpgradeFromVersionOneToVersionTwo(in);
            end
            if in.Version <= 2
                in = iUpgradeFromVersionTwoToVersionThree(in);
            end
            if in.Version <= 3
                in = iUpgradeFromVersionThreeToVersionFour(in);
            end
            internalLayer = nnet.internal.cnn.layer.CrossEntropy(in.Name,...
                in.NumClasses, in.Categories, in.ClassWeights,...
                in.ObservationDim);
            this = nnet.cnn.layer.ClassificationOutputLayer(internalLayer);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            lossFunction = this.LossFunction;
            
            numClasses = numel(this.ClassNames);
            
            if strcmp(this.ClassWeights, 'none')
                wFlag = '';
            else
                wFlag = 'Weighted';
            end
            
            if numClasses==0
                description = iGetMessageString( ...
                    'nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNoClasses', ....
                    lossFunction );
            elseif numClasses==1
                description = iGetMessageString( ...
                    ['nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayOneClass',wFlag], ....
                    lossFunction, ...
                    this.ClassNames{1} );
            elseif numClasses==2
                description = iGetMessageString( ...
                    ['nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayTwoClasses',wFlag], ....
                    lossFunction, ...
                    this.ClassNames{1}, ...
                    this.ClassNames{2} );
            elseif numClasses>=3
                description = iGetMessageString( ...
                    ['nnet_cnn:layer:ClassificationOutputLayer:oneLineDisplayNClasses',wFlag], ....
                    lossFunction, ...
                    this.ClassNames{1}, ...
                    int2str( numClasses-1 ) );
            end
            
            type = iGetMessageString( 'nnet_cnn:layer:ClassificationOutputLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            if numel(this.ClassNames) < 11 && ~ischar(this.Classes)
                propertyList = struct;
                propertyList.Name = this.Name;
                propertyList.Classes = this.Classes';
                propertyList.ClassWeights = this.ClassWeights;
                propertyList.OutputSize = this.OutputSize;
                groups = [
                    matlab.mixin.util.PropertyGroup(propertyList, '');
                    this.propertyGroupHyperparameters( {'LossFunction'} )
                    ];
            else
                generalParameters = {'Name' 'Classes' 'ClassWeights' 'OutputSize'};
                groups = [
                    this.propertyGroupGeneral( generalParameters )
                    this.propertyGroupHyperparameters( {'LossFunction'} )
                    ];
            end
        end
    end
    
    methods(Access = private)
        function this = updateClasses(this, classes)
            name = this.PrivateLayer.Name;
            classWeights = this.PrivateLayer.ClassWeights;
            obsDim = this.PrivateLayer.ObservationDim;
            numClasses = iGetNumClassesFromClasses(classes);
            
            this.PrivateLayer = nnet.internal.cnn.layer.CrossEntropy(name,...
                numClasses, classes, classWeights, obsDim);
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function S = iUpgradeFromVersionOneToVersionTwo( S )
% iUpgradeFromVersionOneToVersionTwo   Add the observation dimension
% property to v1 layers. All v1 layers, created from R2017a and before,
% have observation dimension equal to 4, corresponding to the image data
% format. A NumClasses property is also created for construction of the
% internal layer.
S.Version = 2.0;
S.ObservationDim = 4;
if isempty( S.OutputSize )
    S.NumClasses = S.OutputSize;
else
    S.NumClasses = S.OutputSize(end);
end
end

function S = iUpgradeFromVersionTwoToVersionThree( S )
% iUpgradeFromVersionTwoToVersionThree   In all layers with version <= 2
% the property ClassNames has to be replaced with Categories in 
% version 2. See g1667032. Here set Categories to categorical array with 
% categories as ClassNames and ordinality false.
S.Version = 3.0;
% Make sure there are no duplicate names
classNames = iRenameDuplicated(S.ClassNames);
S.Categories = categorical(classNames, classNames);
end

function S = iUpgradeFromVersionThreeToVersionFour( S )
% iUpgradeFromVersionThreeToVersionFour   In all layers with version <= 3
% the layer does not have a ClassWeights property. 
S.Version = 4.0;
S.ClassWeights = [];
end

function renamed = iRenameDuplicated(names)
    % Makes a list of unique names, avoiding using the names that were
    % duplicated

    % Generate list of duplicated names
    [~,idx] = unique(names);    
    idx = setdiff(1:numel(names), idx);
    if ~isempty(idx)
        % Print warning 
        warning(message('nnet_cnn:layer:ClassificationOutputLayer:RenamingClassNames'));    
        duplicated = unique(names(idx));    
        renamed = matlab.lang.makeUniqueStrings( ...
            names, duplicated(:));
    else
        renamed = names;
    end
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function iAssertValidClasses(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateClasses(value));
end

function iAssertValidClassWeights(value)
functionName = 'ClassificationOutputLayer';
if ischar(value) || isstring(value)
    validatestring(value, {'none'}, functionName, 'ClassWeights');
else
    validateattributes(value, {'numeric'}, ...
        {'nonempty', 'vector', 'positive', 'finite', 'real', 'nonsparse'}, ...
        functionName, 'ClassWeights');
end
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end

function classes = iConvertClassesToCanonicalForm(classes)
if iIsAuto(classes)
    classes = categorical();
else
    classes = ...
        nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(classes);
end
end

function classWeights = iConvertClassWeightsToCanonicalForm(classWeights)
if ischar(classWeights) || isstring(classWeights)
    classWeights = [];
else
    classWeights = reshape(double(classWeights),[],1);
end
end

function iCheckConsistencyClassesAndClassWeights(classes, weights)
if ~isempty(weights)
    if isempty(classes)
        error(message('nnet_cnn:layer:ClassificationOutputLayer:ClassesRequired'));
    elseif numel(weights) ~= numel(classes)
        error(message('nnet_cnn:layer:ClassificationOutputLayer:ClassesWeightsMismatch'));
    end
end
end

function iAddSolutionAndRethrow(exception)
solutionId = "SolutionToModifyClasses";
newId = string(exception.identifier)+solutionId;
solutionMsg = string(message("nnet_cnn:layer:ClassificationOutputLayer:"+solutionId));
newMsg = string(exception.message)+" "+solutionMsg; 
throwAsCaller(MException(newId, newMsg))
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end

function numClasses = iGetNumClassesFromClasses(classes)
if isempty(classes)
    numClasses = [];
else
    numClasses = numel(classes);
end
end

function p = iCreateParser()
p = inputParser;
defaultName = '';
defaultClasses = 'auto';
defaultClassWeights = 'none';
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
addParameter(p, 'Classes', defaultClasses, @iAssertValidClasses);
addParameter(p, 'ClassWeights', defaultClassWeights, @iAssertValidClassWeights); 
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Classes = iConvertClassesToCanonicalForm(p.Results.Classes);
inputArguments.NumClasses = iGetNumClassesFromClasses(inputArguments.Classes);
% make sure strings get converted to char vectors
inputArguments.Name = convertStringsToChars(p.Results.Name); 
inputArguments.ClassWeights = iConvertClassWeightsToCanonicalForm(p.Results.ClassWeights);
end
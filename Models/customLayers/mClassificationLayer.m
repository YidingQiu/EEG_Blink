classdef (Abstract) mClassificationLayer < nnet.cnn.layer.Layer
    % ClassificationLayer   Interface for classification layers

    %   Copyright 2017-2019 The MathWorks, Inc.
    
    properties
        % Name (char vector)   A name for the layer
        Name = '';
        
        % Classes (categorical)  The categories into which the input data
        % is classified.
        %   A categorical column vector whose elements are the distinct
        %   classes to classify the input data to the network. It can be
        %   set passing a string or categorical vector, or a cell vector of
        %   character vectors, or 'auto'. When 'auto' is specified, the
        %   classes are automatically set during training. Default: 'auto'.
        Classes = 'auto';
    end
    
    properties (SetAccess = protected)
        % Description (char vector or scalar string)   A one line 
        %   description for the layer
        Description = ''
        
        % Type (char vector or scalar string)   The type of layer
        Type = ''
    end

    properties (Hidden, Dependent)
        % ClassNames (cellstr)   The names of the classes
        %   A cell array containing the names of the classes.
        ClassNames
    end
    
    properties (Hidden)
        % NumClasses (scalar int)   Number of classes.
        %  If empty, this will be automatically determined at training time.
        NumClasses = [];
    end
        
    methods (Abstract)
        % forwardLoss    Return the loss between the output obtained from
        % the network and the expected output
        %
        % Inputs
        %   this - the output layer to forward the loss through
        %   Y    - Predictions made by network. If backwardLoss it not
        %          overridden, then Y is an unlabelled dlarray and all
        %          operations in forwardLoss must support dlarrays.
        %   T    - Targets (actual values)
        %
        % Outputs
        %   loss - the loss between Y and T. If backwardLoss is not
        %          overridden, then loss must be an unlabelled dlarray.
        loss = forwardLoss( this, Y, T)
    end
    
    methods
        % backwardLoss    Back propagate the derivative of the loss function
        % If backwardLoss is not overridden then automatic differentiation
        % is used to compute derivatives during training.
        %
        % Inputs
        %   this - the output layer to backprop the loss through
        %   Y    - Predictions made by network
        %   T    - Targets (actual values)
        %
        % Outputs
        %   dLdY - the derivative of the loss (L) with respect to the input Y
        function dLdY = backwardLoss( this, Y, T) %#ok<INUSD,STOUT>
            error(message('nnet_cnn:internal:cnn:layer:CustomOutputLayer:CustomlayerUnusedBackwardLoss', class(this)))
        end
    end
    
    methods(Hidden, Access = protected)
        function layer = mClassificationLayer()
            try
                nnet.internal.cnn.layer.util.CustomLayerVerifier.validateMethodSignatures(layer);
            catch e
                % Wrap exception in a CNNException, which reports the error in a custom way
                err = nnet.internal.cnn.util.CNNException.hBuildCustomError( e );
                throwAsCaller(err);
            end
        end

        function [description, type] = getOneLineDisplay( layer )
            if strlength( layer.Description ) == 0
                description = iGetMessageString( 'nnet_cnn:layer:ClassificationLayer:oneLineDisplay' );
            else
                description = layer.Description;
            end
            
            if strlength( layer.Type ) == 0
                type = iGetMessageString( 'nnet_cnn:layer:ClassificationLayer:Type' );
            else
                type = layer.Type;
            end
        end
    end
    
    methods
        function layer = set.Name( layer, val )
            iAssertValidLayerName( val );
            layer.Name = convertStringsToChars( val );
        end
        
        function layer = set.ClassNames( layer, val )
            layer.Classes = val;
        end
        
        function val = get.ClassNames( layer )
            if iIsAuto(layer.Classes)
                val = {};
            else
                val = categories(layer.Classes);
            end
        end
        
        function val = get.Classes( layer )
            val = layer.Classes;
        end
        
        function layer = set.Classes( layer, val )
            iAssertValidClasses(val);
            if iIsAuto(val)
                layer = layer.resetNumClasses();
                layer.Classes = 'auto';
            else                
                classes = iConvertClassesToCanonicalForm(val); 
                layer = layer.inferSizeFromClasses(classes);
                layer.Classes = classes;
            end
        end
        
        function layer = set.Description( layer, val )
            val = iValidateText( val, "Description" );
            layer.Description = val;
        end
        
        function layer = set.Type( layer, val )
            val = iValidateText( val, "Type" );
            layer.Type = val;
        end
    end
        
    methods(Access = private) 
        function layer = inferSizeFromClasses(layer, classes)           
            layer.NumClasses = numel(classes);
        end
        
        function layer = resetNumClasses(layer)           
            layer.NumClasses = [];
        end
    end
end

function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end

function iAssertValidLayerName( name )
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName( name ));
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end

function iAssertValidClasses(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateClasses(value));
end

function classes = iConvertClassesToCanonicalForm(classes)
classes = ...
    nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(classes);
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end

function paramValue = iValidateText( paramValue, paramName )
isValidText = nnet.internal.cnn.layer.paramvalidation.isValidStringOrCharArray(paramValue);
if isempty(paramValue)
    paramValue = '';
elseif ~isValidText
    error(message('nnet_cnn:layer:Layer:ParamMustBeStringOrChar',paramName))
end
end
classdef gatherLayer < nnet.layer.Layer % ...
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.
        
        % Declare layer properties here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
    end

    properties (State)
        % (Optional) Layer state parameters.
%         NumInputs
        numGather
        CatDim
        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.
        memory
        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = gatherLayer(numGather,layerName)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.numGather = numGather;
            layer.Name = layerName;
            layer.CatDim = 2;
%             layer.NumInputs = 512;
            layer.Description = "pile up seperate S slices";
            % Define layer constructor function here.
        end

        function [Z] = predict(layer,varargin)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            X = varargin;
            outSize = [1 1 size(X{1},3) 1];
            Z = ones(outSize);
            for i = 1:nargin
                Y = extractdata(X{i});
                Z = cat(layer.CatDim,Z,Y);
            end
            Z = Z(:,2:end,:,:);

        end


        end
    
end
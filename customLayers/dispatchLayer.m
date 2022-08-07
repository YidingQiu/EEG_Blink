classdef dispatchLayer < nnet.layer.Layer % ...
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

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = dispatchLayer(numDispatch,layerName)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            layer.Name = layerName;
            
            
            layer.Description = "cut up S dimension into slices";
            % Define layer constructor function here.
        end

        function [varargout] = predict(layer,X)
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

            outSize = size(X{1},2);
            Y = extractdata(X);
            for i = 1:outSize
                Yi = Y(:,i,:,:);
                varargout{i} = dlarray(Yi,"SSCB");
            end
            
        end


        end
    
end
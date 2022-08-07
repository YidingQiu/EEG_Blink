classdef transposeLayer < nnet.layer.Layer & nnet.layer.Formattable
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
        % parameters and state.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = transposeLayer()
        %%layer
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = "transpose";
            
            layer.Description = "Transpose";

            % Define layer constructor function here.
        end

        function [Z] = predict(layer,X)

            Z = extractdata(dlarray(X));

            
            Z = permute(Z,[1 4 2 3]);%%transpose
%             Z = permute(Z,[3 4 2 1]);
%             Z = dlarray(reshape(single(Z),size(Z,[1:3])),"CBT");
            Z = dlarray(Z, "SSCB");
        end
      end
end


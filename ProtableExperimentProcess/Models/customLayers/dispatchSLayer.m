classdef dispatchSLayer < nnet.layer.Layer & nnet.layer.Formattable
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
        function layer = dispatchSLayer()
        %%layer
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = "dispatchSLayer";
            
            layer.Description = "send S dimension to T";

            % Define layer constructor function here.
        end

        function [Z] = predict(layer,X)
            % disp('------------')
            % get initial size
            B=size(X,4);C=size(X,3);T=size(X,2);
            % start a template for output
            Z = ones([C B 1]);
            Y = extractdata(X);

            for i = 1:T
                % concatenate at T dimension
                Z = cat(3,Z,reshape(Y(:,i,:),[C B 1]));
%                 disp(size(Z));
            end
%             disp(size(Z));
            % remove the template one
            Z = Z(:,:,2:end);
%             disp(size(Z));               
            Z = dlarray(Z,"CBT");
        end
    end

    methods(Access=private)
        %function 

    
    
    
    end

end


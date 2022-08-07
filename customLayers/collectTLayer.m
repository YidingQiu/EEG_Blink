classdef collectTLayer < nnet.layer.Layer & nnet.layer.Formattable
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
        function layer = collectTLayer()
        %%layer
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = "collectTLayer";
            
            layer.Description = "pack T dimension into S";

            % Define layer constructor function here.
        end

        function [Z] = predict(layer,X)
% %             Bdim = find(X,dim(X) == 'B');
% %             Cdim = find(X,dim(X) == 'C');
% %             Tdim = find(dims(X) == 'T');
% %             firstSdim = find(dims(X) == 'S',1);
%             Z = extractdata(dlarray(X));
% 
% %             Z = permute(Z,[firstSdim Tdim Cdim Bdim]);
%             Z = permute(Z,[1 4 2 3]);
%             Z = dlarray(Z,"CUTB");
%             disp('    ----------');
%             disp(size(X));
            Y = extractdata(X);
%             disp(size(Y));
            C = size(X,1);
            B=size(X,2);T=size(X,3);
            shape = [1 1 C B];%[C 1 1 B]
            Z = ones(shape);
%             disp(B);
            for i = 1:T

                %Z = cat(2,Z,reshape(Y(:,:,i),[C 1 1 B]));
                Z = cat(2,Z,reshape(Y(:,:,i),shape));
            end
            Z = Z(:,2:end,:,:);
            Z = dlarray(Z,"SSCB");

%             disp([size(Z) 'Z']);


        end
    end

    methods(Access=private)
        %function 

    
    
    
    end

end


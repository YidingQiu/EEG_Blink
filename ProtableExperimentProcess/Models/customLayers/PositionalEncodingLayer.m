classdef PositionalEncodingLayer < nnet.layer.Layer
    methods
        function layer = PositionalEncodingLayer(name)
            layer.Name = name;
            layer.Description = "Positional Encoding Layer";
        end
        
        function Z = predict(layer, X)
            % Get dimensions
            [sequenceLength, modelDim, batchSize] = size(X);
            
            % Calculate positional encoding
            posEnc = zeros(sequenceLength, modelDim);
            for pos = 1:sequenceLength
                for i = 0:(modelDim/2 - 1)
                    posEnc(pos, 2*i + 1) = sin(pos / (10000^(2*i / modelDim)));
                    posEnc(pos, 2*i + 2) = cos(pos / (10000^(2*i / modelDim)));
                end
            end
            posEnc = permute(posEnc, [1 3 2]);
            posEnc = repmat(posEnc, 1, batchSize, 1);
            
            % Add positional encoding to the input
            Z = X + posEnc;
        end
    end
end

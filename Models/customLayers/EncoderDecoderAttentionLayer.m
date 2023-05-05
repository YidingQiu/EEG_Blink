classdef EncoderDecoderAttentionLayer < nnet.layer.Layer
    properties
        numHiddenUnits
    end
    
    methods
        function layer = EncoderDecoderAttentionLayer(name, numHiddenUnits)
            layer.Name = name;
            layer.numHiddenUnits = numHiddenUnits;
            layer.Description = "Attention Layer with " + numHiddenUnits + " hidden units";
        end
        
        function Z = predict(layer, X)
            % X{1} - Encoder hidden states (sequence length x batch size x hidden units)
            % X{2} - Decoder hidden state (1 x batch size x hidden units)
            
            encoderStates = X{1};
            decoderState = X{2};
            
            % Compute attention scores
            scores = dot(encoderStates, decoderState, 3);
            
            % Compute attention weights (softmax)
            weights = exp(scores) ./ sum(exp(scores), 1);
            
            % Compute context vector
            context = sum(encoderStates .* weights, 1);
            
            % Combine context vector with decoder state
            Z = cat(1, context, decoderState);
        end
        
        function dLdX = backward(layer, X, ~, dLdZ, ~)
            % X{1} - Encoder hidden states (sequence length x batch size x hidden units)
            % X{2} - Decoder hidden state (1 x batch size x hidden units)
            
            encoderStates = X{1};
            decoderState = X{2};
            
            % Compute attention scores
            scores = dot(encoderStates, decoderState, 3);
            
            % Compute attention weights (softmax)
            weights = exp(scores) ./ sum(exp(scores), 1);
            
            % Backpropagate through attention mechanism
            dLdX = cell(1, 2);
            dLdX{1} = []; % Gradient for encoderStates is not needed in this example
            dLdX{2} = sum(dLdZ, 1);
        end
    end
end

classdef MultiHeadAttentionLayer < nnet.layer.Layer
    properties
        numHeads;
        dModel;
        dK;
    end
    
    methods
        function layer = MultiHeadAttentionLayer(numHeads, dModel, name)
            layer.Name = name;
            layer.Description = "Multi-head attention layer with " + numHeads + " heads";
            
            layer.numHeads = numHeads;
            layer.dModel = dModel;
            layer.dK = dModel / numHeads;
        end
        
        function Z = predict(layer, X)
            % X: Cell array containing queries, keys, and values.
            queries = X{1};
            keys = X{2};
            values = X{3};
            
            % Calculate attention scores
            scores = (queries * keys') / sqrt(layer.dK);
            
            % Softmax normalization
            attentionWeights = softmax(scores, 2);
            
            % Calculate output
            Z = attentionWeights * values;
        end
        
        function dLdX = backward(layer, X, Z, dLdZ, memory)
            queries = X{1};
            keys = X{2};
            values = X{3};
            
            scores = (queries * keys') / sqrt(layer.dK);
            attentionWeights = softmax(scores, 2);
            
            % Gradients
            dLdQ = dLdZ * keys;
            dLdK = (dLdZ' * queries)';
            dLdV = attentionWeights' * dLdZ;
            
            dLdX = {dLdQ, dLdK, dLdV};
        end
    end
end

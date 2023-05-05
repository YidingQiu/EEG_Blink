classdef MultiHeadSelfAttentionLayer < nnet.layer.Layer
    properties
        NumHeads
        ModelDim
        KeyDim
        ValueDim
        Wq
        Wk
        Wv
        Wo
    end
    
    methods
        function layer = MultiHeadSelfAttentionLayer(numHeads, modelDim, name)
            layer.Name = name;
            layer.Description = "Multi-Head Self-Attention Layer";
            
            layer.NumHeads = numHeads;
            layer.ModelDim = modelDim;
            layer.KeyDim = modelDim / numHeads;
            layer.ValueDim = modelDim / numHeads;
            
            layer.Wq = randn(modelDim, modelDim, 'single') * 0.1;
            layer.Wk = randn(modelDim, modelDim, 'single') * 0.1;
            layer.Wv = randn(modelDim, modelDim, 'single') * 0.1;
            layer.Wo = randn(modelDim, modelDim, 'single') * 0.1;
        end
        
        function Z = predict(layer, X)
            % Get dimensions
            [sequenceLength, ~, batchSize] = size(X);
            
            % Compute query, key, and value
            query = X * layer.Wq;
            key = X * layer.Wk;
            value = X * layer.Wv;
            
            % Split into multiple heads
            query = reshape(query, sequenceLength, layer.KeyDim, layer.NumHeads, batchSize);
            key = reshape(key, sequenceLength, layer.KeyDim, layer.NumHeads, batchSize);
            value = reshape(value, sequenceLength, layer.ValueDim, layer.NumHeads, batchSize);
            
            % Permute dimensions for easier computation
            query = permute(query, [1 3 2 4]);
            key = permute(key, [1 3 2 4]);
            value = permute(value, [1 3 2 4]);
            
            % Compute scaled dot-product attention
                        scores = dot(query, key, 3) ./ sqrt(layer.KeyDim);
            attentionWeights = softmax(scores, 1);
            context = dot(attentionWeights, value, 1);
            
            % Combine heads
            context = permute(context, [1 3 2 4]);
            context = reshape(context, sequenceLength, layer.ModelDim, batchSize);
            
            % Apply output linear projection
            Z = context * layer.Wo;
        end
    end
end
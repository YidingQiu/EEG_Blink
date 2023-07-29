classdef TransformerLayer < nnet.layer.Layer
    properties
        NumHeads
        ModelDim
        FfDim
        Attention
        Norm1
        Dropout1
        Fc1
        Fc2
        Norm2
        Dropout2
    end
    
    methods
        function layer = TransformerLayer(numHeads, modelDim, ffDim, name, dropout)
            layer.Name = name;
            layer.Description = "Transformer Layer";
            
            layer.NumHeads = numHeads;
            layer.ModelDim = modelDim;
            layer.FfDim = ffDim;
            
            layer.Attention = MultiHeadSelfAttentionLayer(numHeads, modelDim, 'attention');
            layer.Norm1 = layerNormalizationLayer('Name', 'norm1');
            layer.Dropout1 = dropoutLayer(dropout, 'Name', 'dropout1');
            layer.Fc1 = fullyConnectedLayer(ffDim, 'Name', 'fc1');
            layer.Fc2 = fullyConnectedLayer(modelDim, 'Name', 'fc2');
            layer.Norm2 = layerNormalizationLayer('Name', 'norm2');
            layer.Dropout2 = dropoutLayer(dropout, 'Name', 'dropout2');
        end
        
        function Z = predict(layer, X)
            % Multi-head self-attention
            attentionOut = layer.Attention.predict(X);
            
            % Add and norm
            X = layer.Norm1.predict(X + layer.Dropout1.predict(attentionOut));
            
            % Feed-forward
            ffOut = layer.Fc2.predict(relu(layer.Fc1.predict(X)));
            
            % Add and norm
            Z = layer.Norm2.predict(X + layer.Dropout2.predict(ffOut));
        end
    end
end

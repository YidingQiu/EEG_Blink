function lgraph = standardBranch(convolution, branch_id, block_id, recSize, filterNum)
    % Create a layerGraph and add a dummy input layer
    lgraph = layerGraph(functionLayer(@(x) x, "Name","block_in_" + branch_id + "_" + block_id));
    % Apply convolution to each channel separately
    for channel = 1:filterNum
        lgraph = addLayers(lgraph,...
            functionLayer(@(x) x(:, :, :, channel, :),...
            Name="channel_in_" + channel + "_" + branch_id + "_" + block_id));
        lgraph = addLayers(lgraph,...
            convolution(recSize,...
            "filters_" + channel + "_" + branch_id + "_" + block_id));
        lgraph = connectLayers(lgraph,...
            "block_in_" + branch_id + "_" + block_id,...
            "channel_in_" + channel + "_" + branch_id + "_" + block_id);
        lgraph = connectLayers(lgraph,...
            "channel_in_" + channel + "_" + branch_id + "_" + block_id,...
            "filters_" + channel + "_" + branch_id + "_" + block_id);
    end
   
    % Concatenate results of convolution
    lgraph = addLayers(lgraph,...
        depthConcatenationLayer(filterNum,...
        Name="depth_concat_in_" + branch_id + "_" + block_id));
    for channel = 1:filterNum
        lgraph = connectLayers(lgraph,...
            "filters_" + channel + "_" + branch_id + "_" + block_id,...
            "depth_concat_in_" + branch_id + "_" + block_id + "/in" + channel);
    end
    % Add normalization and activation
    layers = [...
        instanceNormalizationLayer(...
            Name="norm_" +  branch_id + "_" + block_id,...
            OffsetLearnRateFactor=0,...
            OffsetInitializer="zeros")... % No offset
        functionLayer(@(x) max(tanh(x), 0),... 
            Name="activation_" +  branch_id + "_" + block_id)...
        functionLayer(@(x) x,...
            Name="block_out_" + branch_id + "_" + block_id)...
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph,...
        "depth_concat_in_" + branch_id + "_" + block_id,...
        "norm_" +  branch_id + "_" + block_id);
   
end
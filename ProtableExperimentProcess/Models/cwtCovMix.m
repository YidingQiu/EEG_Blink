function lgraph = cwtCovMix(opts)
% convMixerLayers   Build ConvMixer architecture.

    arguments
        opts.numFeatures = 3;
        opts.patchSize = 5;
        opts.Depth = 2;
        opts.HiddenDimension = 4;
        opts.filterSizeS = 3;
        opts.response = 3;
        opts.paddingSizeS = 1;
        opts.dropoutLayerRegularization = 0.5;
    end
    
    numFeatures = opts.numFeatures;
    patchSize = [opts.patchSize 1];
    depth = opts.Depth;
    hidden_dim = opts.HiddenDimension;
    filterSizeS = opts.filterSizeS;
    response = opts.response;
    paddingSizeS = opts.paddingSizeS;
    dropoutLayerRegularization = opts.dropoutLayerRegularization;
    
    % Input Layers
    inputLayers = [
        sequenceInputLayer(numFeatures,'Name','input',"MinLength",512,"Normalization","zscore")
        cwtLayer("Name","cwt","SignalLength",512,"VoicesPerOctave",48,"FrequencyLimits",[0.0026041 0.3125])
        
        convolution2dLayer(opts.patchSize, hidden_dim, ...
            Stride=patchSize, ...
            Name="patchEmbedding", ...
            WeightsInitializer="glorot", ...
            Padding="same")
        geluLayer(Name="gelu_0")
        batchNormalizationLayer(Name="batchnorm_0")];
    
    % Make Layer Graph
    lgraph = layerGraph(inputLayers);
    
    for i = 1:depth
    
        sublayer = channelSubset(1:hidden_dim,i);
        lgraph = addLayers(lgraph,sublayer);
        depthConcate = depthConcatenationLayer(hidden_dim,'Name',"concate"+i);
        lgraph = addLayers(lgraph,depthConcate);
        
        for j = 1:hidden_dim
            
            lgraph = connectLayers(lgraph,"Group"+i+"_"+j,"concate"+i+"/in"+j);
            if j < hidden_dim    
                lgraph = disconnectLayers(lgraph,"Group"+i+"_"+j,"Group"+i+"_"+(j+1));
            end
        end
        
        convMixer = [
            geluLayer(Name="gelu_"+(2*i-1))
            batchNormalizationLayer(Name="batchnorm_"+(2*i-1))
            additionLayer(2,Name="addition_"+i)
            convolution2dLayer([1 1],hidden_dim,Name="pointwiseConv_"+i,WeightsInitializer="glorot")
            geluLayer(Name="gelu_"+2*i)
            batchNormalizationLayer(Name="batchnorm_"+2*i)];
        lgraph = addLayers(lgraph,convMixer);    
        for j = 1:hidden_dim
            lgraph = connectLayers(lgraph,"batchnorm_"+(2*i-2),"Group"+i+"_"+j);
        end
    
    
        lgraph = connectLayers(lgraph,"concate"+i,"gelu_"+(2*i-1));
    
        %lgraph = connectLayers(lgraph,"batchnorm_"+2*(i-1),"concate"+i);
        lgraph = connectLayers(lgraph,"batchnorm_"+(2*(i-1)),"addition_"+i+"/in2");
        
        %lgraph = connectLayers(lgraph,"","");
    end
    
    % Output Layers
    outputLayers = [ 
        convolution2dLayer([floor(269/opts.patchSize)+1 filterSizeS],response,'Padding',[0 0 floor(filterSizeS/2) floor(filterSizeS/2)],"Name",'fullCov')
    
        dropoutLayer(dropoutLayerRegularization,'Name','Dropout')
        softmaxLayer('Name','softMax')
        classificationLayer("Classes",'auto')];
    
    lgraph = addLayers(lgraph,outputLayers);
    lgraph = connectLayers(lgraph,"batchnorm_"+2*depth,"fullCov");
end



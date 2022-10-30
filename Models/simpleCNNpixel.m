function lgraph = simpleCNNpixel(imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization,classes,activation)
    arguments
    imSize,
    numBlocks = 3,
    filterSizeC=5,
    filterSizeS =5,
    filterNum=16, 
    dropoutLayerRegularization=0.05,
    classes = ["blink","noBlink", "muscleArtifact"],
    activation = 'relu'
    end

    switch activation
        case 'tanh'
            activationFunction = tanhLayer();
        case 'swish'
            activationFunction = swishLayer();
        otherwise
            activationFunction = reluLayer();
    end
    
    paddingSizeS = floor(filterSizeS/2);
    paddingSizeC = floor(filterSizeC/2);
    pool = floor(nthroot(imSize(1),numBlocks))+2;    
    response = 3;
    
    layers = imageInputLayer(imSize,'Name','input');
    lgraph = layerGraph(layers);
    outputName = layers.Name;
    remains = imSize(1); % size of first S channel
    
    for i = 1:numBlocks
        
        % get size of pooling
        if remains <= pool || i == numBlocks
            pool = remains;
        end
        remains = floor(imSize(1)/pool)+1;
        paddingSizeP = floor(pool/2);
        % block
        layers = [               
            convolution2dLayer([filterSizeC filterSizeS], filterNum, ...
            'Padding',[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],'Name',"cov_"+i)

            batchNormalizationLayer
            activationFunction

            maxPooling2dLayer([pool 1],'Stride',[pool 1], ...
            'Padding',[paddingSizeP paddingSizeP 0 0],'Name',"pool_"+i)   
        %     averagePooling2dLayer
            ];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"cov_"+i);
        
        outputName = "pool_"+i;
        
    end  
    
    %     %%%%%%%%%%%
    %     functionLayer(@(X) dlarray(X,"SCSB"),Formattable=true,Description="channel to spatial")
    %     convolution2dLayer([1 64],512,'Padding',[0,0,2,0],'PaddingValue',0,"Name",'fullCov')
    %     functionLayer(@(X) dlarray(X,"SCSB"),Formattable=true,Description="spatial to channel ")
    % %%%%%%%%%%%%%%%%
    % convolution2dLayer([1 1],3,"Name",'fullCov')
    
    layers = [ 
        convolution2dLayer([1 filterSizeS],response,'Padding',[0 0 paddingSizeS paddingSizeS],"Name",'fullCov')
        dropoutLayer(dropoutLayerRegularization,'Name','Dropout')
        softmaxLayer('Name','softMax')
        pixelClassificationLayer("Classes",classes)];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'fullCov');


    
    
    % lgraph = layerGraph(layers);
    % 
    
    
    
    
    % for i = 2:slice
    % % https://au.mathworks.com/help/deeplearning/ug/define-custom-layer-with-multiple-inputs.html
    %     %dispatchLayer(i,'dispatch_'+i)
    %     disp(i);
    %     fcName = ['fc_' i];
    %     fullyConnected = [fullyConnectedLayer(4,'Name',fcName)];
    %     lgraph = addLayers(lgraph,fullyConnected);
    %     lgraph = connectLayers(lgraph,'dispatch',fcName);
    %     lgraph = connectLayers(lgraph,fcName,'gather');
    %     
    % end
    
    
    
                    
    
                    
    


end

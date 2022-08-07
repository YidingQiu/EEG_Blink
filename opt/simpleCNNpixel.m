function lgraph = simpleCNNpixel(imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization)
    arguments
    imSize,
    numBlocks = 3,
    filterSizeC=5,
    filterSizeS =5,
    filterNum=16, 
    dropoutLayerRegularization=0.05
    end
    
    paddingSizeS = floor(filterSizeS/2);
    paddingSizeC = floor(filterSizeC/2);

    pool = floor(nthroot(imSize(1),numBlocks))+1;
    
    response = 3;
    
    layers = imageInputLayer(imSize,'Name','input');
    lgraph = layerGraph(layers);
    outputName = layers.Name;
    remains = imSize(1);
    
    for i = 1:numBlocks
        
        if remains <= pool || i == numBlocks
            pool = remains;
        end
        
        remains = floor(imSize(1)/pool)+1;
        paddingSizeP = floor(pool/2);
        layers = [               
            convolution2dLayer([filterSizeC filterSizeS], filterNum, ...
            'Padding',[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],'Name',"cov_"+i)
            batchNormalizationLayer
            reluLayer()           
            maxPooling2dLayer([pool 1],'Stride',[pool 1], ...
            'Padding',[paddingSizeP paddingSizeP 0 0],'Name',"pool_"+i)   
        %     averagePooling2dLayer
            ];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"cov_"+i);
        
        outputName = "pool_"+i;
        
    end
    %     convolution2dLayer([filterSizeC filterSizeS], filterNum, ...
    %     'Padding',[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],'Name','cov2')
    %     batchNormalizationLayer
    %     reluLayer
    %     maxPooling2dLayer([pool 1],'Stride',[pool 1],'Name','pool2')   
    % %     averagePooling2dLayer([4 1],'Stride',[4 1],'Name','pool2')  
    % 
    %     convolution2dLayer([filterSizeC filterSizeS], filterNum, ...
    %     'Padding',[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],'Name','cov3')
    %     batchNormalizationLayer
    %     reluLayer     
    %     maxPooling2dLayer([pool 1],'Stride',[pool 1],'Name','pool3')   
    % %     averagePooling2dLayer([4 1],'Stride',[4 1],'Name','pool3')  
    
    
    
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
        pixelClassificationLayer()];                
    
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

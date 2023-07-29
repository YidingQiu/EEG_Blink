function lgraph = CWTCovMixer(numFeatures,varargin)
    
    p = inputParser;
    
    % Define default values for optional arguments.
    defaultNumBlocks = 3;
    defaultFilterSizeC = 5;
    defaultFilterSizeS = 5;
    defaultFilterNum = 16;
    defaultHiddenDimension = 64;
    defaultDropoutLayerRegularization = 0.05;
    defaultClasses = ["blink","n/a", "muscle-artifact","no-blink"];
    defaultActivation = 'relu';
    
    % Add required and optional args.
    addRequired(p, 'numFeatures');
    addOptional(p, 'numBlocks', defaultNumBlocks);
    addOptional(p, 'filterSizeC', defaultFilterSizeC);
    addOptional(p, 'filterSizeS', defaultFilterSizeS);
    addOptional(p, 'filterNum', defaultFilterNum);
    addOptional(p, 'hiddenDimension', defaultHiddenDimension);
    addOptional(p, 'dropoutLayerRegularization', defaultDropoutLayerRegularization);
    addOptional(p, 'classes', defaultClasses);
    addOptional(p, 'activation', defaultActivation);
    
    % Parse input args.
    parse(p, numFeatures, varargin{:});
    
    % Assign values in your function from the parsed input.
    numBlocks = p.Results.numBlocks;
    filterSizeC = p.Results.filterSizeC;
    filterSizeS = p.Results.filterSizeS;
    filterNum = p.Results.filterNum;
    hiddenDimension = p.Results.hiddenDimension;
    dropoutLayerRegularization = p.Results.dropoutLayerRegularization;
    classes = p.Results.classes;
    activation = p.Results.activation;

    switch activation
        case 'tanh'
            activationFunction = tanhLayer();
        case 'swish'
            activationFunction = swishLayer();
        otherwise
            activationFunction = reluLayer();
    end
    

    layers = [
        sequenceInputLayer(numFeatures,'Name','input',"MinLength",512)
        cwtLayer("Name","cwt","SignalLength",512,"VoicesPerOctave",48,"FrequencyLimits",[0.0026041 0.3125])
        ];%batchNormalizationLayer(Name="batchnorm_0")
    lgraph = layerGraph(layers);
    outputName = "cwt";

    inputChannel = 3;
    numFeatures = 269; % get this
    paddingSizeS = floor(filterSizeS/2);
    paddingSizeC = floor(filterSizeC/2);
    pool = floor(nthroot(numFeatures,numBlocks))+2;    
    response = numel(classes);
    

    remains = numFeatures; % size of first S channel

    
    for i = 1:numBlocks
        if remains <= pool || i == numBlocks
            pool = remains;
        end
        remains = floor(numFeatures/pool)+1;
        paddingSizeP = floor(pool/2);
        kernel_size = [filterSizeC filterSizeS];
        convMixer = [
            convolution2dLayer(kernel_size,inputChannel,Name="depthwiseConv_"+i, ...
            Padding=[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],WeightsInitializer="glorot")
            geluLayer(Name="gelu_"+(2*i-1))

            batchNormalizationLayer(Name="batchnorm_"+(2*i-1))

            additionLayer(2,Name="addition_"+i)
            maxPooling2dLayer([pool 1],'Stride',[pool 1], ...
            'Padding',[paddingSizeP paddingSizeP 0 0],'Name',"pool_"+i)   
            convolution2dLayer([1 1],inputChannel,Name="pointwiseConv_"+i,WeightsInitializer="glorot")
            geluLayer(Name="gelu_"+2*i)

            batchNormalizationLayer(Name="batchnorm_"+2*i)
            ];
        lgraph = addLayers(lgraph,convMixer);
        if i == 1
            lgraph = connectLayers(lgraph,"cwt","depthwiseConv_"+i);
            lgraph = connectLayers(lgraph,"cwt","addition_"+i+"/in2");
        else
            lgraph = connectLayers(lgraph,"batchnorm_"+2*(i-1),"depthwiseConv_"+i);
            lgraph = connectLayers(lgraph,"batchnorm_"+2*(i-1),"addition_"+i+"/in2");
        end
        
        
    end

    outputName = "batchnorm_"+2*numBlocks;
    
    layers = [ 
        convolution2dLayer([1 filterSizeS],response,'Padding',[0 0 paddingSizeS paddingSizeS],"Name",'fullCov')
        dropoutLayer(dropoutLayerRegularization,'Name','Dropout')
        softmaxLayer('Name','softMax')
        classificationLayer("Classes",'auto')];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'fullCov');



end

function [dsTrain,dsTest] = imgDsPreparation()

%% train ds

    YLable = load(fullfile('DataSet','HyperparameterSearch','Test','YSearchTest.mat')).YSearchTest;
    dataClassNames = ["blink","n/a", "muscle-artifact"];pixelLabelIds = 1:numel(dataClassNames);
    classNames=["blink","noBlink", "muscleArtifact"];
    imdsTrain = imageDatastore(fullfile('DataSet','HyperparameterSearch','Train','Img','WTImg'),FileExtensions=".jpg");
    pxdsTrain = pixelLabelDatastore(fullfile('DataSet','HyperparameterSearch','Train','Img','PLImg'),classNames,pixelLabelIds);
    
    imdsTest = imageDatastore(fullfile('DataSet','HyperparameterSearch','Test','Img','WTImg'),FileExtensions=".jpg");
    pxdsTest = pixelLabelDatastore(fullfile('DataSet','HyperparameterSearch','Test','Img','PLImg'),classNames,pixelLabelIds);
    
    dsTrain = combine(imdsTrain,pxdsTrain);
%% test ds

    testImgSize = [size(imread(imdsTest.Files{1,1})) 1 1];
    imgsTest = ones(testImgSize);
    for i =1:numel(imdsTest.Files)
        imgsTest = cat(4,imgsTest,reshape(imread(imdsTest.Files{i,1}), testImgSize));
    end
    imgsTest=imgsTest(:,:,:,2:end);
    
    TY = ones([1 512 3 1]);
    
    for i = 1:numel(YLable)   
        YPiece1 = zeros([1 512 1 1]);YPiece2=YPiece1;YPiece3=YPiece1;
        % {"blink","noBlink", "muscleArtifact"}
        YPiece = YLable{1,i};
        YPiece1 = reshape((YPiece=='blink'),[1 512 1 1]);
        YPiece2 = reshape((YPiece=='no-blink'),[1 512 1 1]);
        YPiece2 = YPiece2+reshape((YPiece=='n/a'),[1 512 1 1]);
        YPiece3 = reshape((YPiece=='muscle-artifact'),[1 512 1 1]);
        YPiece = cat(3,YPiece1,YPiece2);YPiece = cat(3,YPiece,YPiece3);
    
        TY = cat(4,TY,YPiece);
    end
    TY = TY(:,:,:,2:end);
    dsTest = {TY,imgsTest};

end
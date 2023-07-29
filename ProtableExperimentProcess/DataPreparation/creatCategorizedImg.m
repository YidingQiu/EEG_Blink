function []=creatCategorizedImg(chopedData, path,slice,dataClassNames)
    arguments
        chopedData,
        path,
        slice = 512,
        dataClassNames = ["blink","n/a", "muscle-artifact"]
    end
    originPath = path;
    path = fullfile(path,'PLImg');
    if exist(path)==0
        mkdir(path);
    end


%     cellStore = {};
    pixelLabelIds = 1:numel(dataClassNames);
    nameLength = length(num2str(size(chopedData,2)));
    for i = 1:numel(chopedData)
        lable = chopedData{1,i};
        firstName = num2str(i);
        while length(firstName)<nameLength
            firstName = ['0' firstName];
        end
        lableData = zeros([1 slice],'int8');
        for j = pixelLabelIds

            lableData(lable==dataClassNames(j)) = j;
        end
        %erase 0 to 2(noblink)
        lableData(lableData==0)=2;
%         cellStore{end+1} = lableData;
        imFileName = strcat(firstName, "_pl.jpg");
        imwrite(uint8(lableData),fullfile(path,imFileName));%pixel lable
    end
%     save(fullfile(originPath,"plImg.mat"),"cellStore","-mat")

    %disp('pixel lable done');
end
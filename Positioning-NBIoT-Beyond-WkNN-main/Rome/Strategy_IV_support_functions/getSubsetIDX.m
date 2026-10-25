function logicalArray = getSubsetIDX(trainingSet_UTM, coarseCoordsTest, threshold)

    %trainingSet = utmConversion(trainingSet);
    coordsTrain = cell2mat(trainingSet_UTM(:, 1:2));

    distances = pdist2(coordsTrain, coarseCoordsTest);
    logicalArray = distances < threshold;

end
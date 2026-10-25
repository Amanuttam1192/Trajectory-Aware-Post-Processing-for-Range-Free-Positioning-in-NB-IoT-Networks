function coarseCoordsTest = coarsePosition(XTest, models, centers)
    
    num_clusters = length(models);
    YPred = zeros(size(XTest, 1), num_clusters);

    for i = 1:length(models)
        YPred (:, i) =  predict(models{i}, XTest);
    end
    
    coarseCoordsTest = zeros(size(XTest, 1), 2);    

    for j = 1: size(XTest, 1)
        coarseCoordsTest (j,:) = estimate_position_RF(YPred(j,:), centers);
    end
    

end
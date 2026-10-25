function [models] = trainRanForestFCM(X, Y)
    % Random Forest Parameters
    num_clusters = size(Y, 2);
    n_estimators = 200; % Number of estimators
    max_depth = 30; % Maximum depth of trees
    %min_samples_split = 2; % Minimum number of samples to split a node
    min_samples_leaf = 4; % Minimum number of samples per leaf

    models = cell(1, num_clusters);
    for i = 1:num_clusters
        %fprintf('Training model %d/%d...\n', i, num_clusters)
        models{i} = TreeBagger(n_estimators, X, Y(:, i), ...
            'Method', 'regression', ...
            'MaxNumSplits', max_depth, ...
            'MinLeafSize', min_samples_leaf, ...
            'OOBPrediction', 'on');
    end


end
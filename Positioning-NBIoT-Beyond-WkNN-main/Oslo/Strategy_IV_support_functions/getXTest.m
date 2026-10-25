function X = getXTest(trainSet, testSet, featureID, bounds, pciVocabulary)
    minRefValue = bounds(1); % reported value for missing PCI
    maxRefValue = bounds(2); % maximum value for PCI, useful in thesis to remove outliers
    removeINF = false;
    normalize = false;

    testSet = applyVocabulary(testSet, pciVocabulary);

    num_points = size(testSet, 1);
    unique_pcis = unique(cell2mat(cellfun(@(x) x(:, 1), trainSet(:, 3), 'UniformOutput', false)));
    
    X = -inf(num_points, length(unique_pcis));
    for i = 1:num_points
        X(i, :) = format_entry_to_row(testSet(i, :), unique_pcis, featureID); % flat array for given parameter      
    end
    isFinite = isfinite(X);
    
    % apply bounds
    X(isFinite & X < minRefValue) = minRefValue;
    X(isFinite & X > maxRefValue) = maxRefValue;
    if removeINF ==true
        X(isinf(X) & X < 0) = minRefValue;    
    end
    if normalize == true
        finite_values = X(~isinf(X));
        max_value = max(finite_values);
        min_value = min(finite_values);
        finite_mask = ~isinf(X);
        X(finite_mask) = (X(finite_mask) - min_value) / (max_value - min_value);
    end

end
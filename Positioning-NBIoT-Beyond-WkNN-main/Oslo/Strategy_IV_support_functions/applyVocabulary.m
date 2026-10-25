function newDataSet = applyVocabulary(dataSet, pciVocabulary)
    %Function to apply an existing vocabulary to a dataset without one. The
    %dataset is expected not to include any PCI not present in the
    %vocabulary.
    %nOriginalMatrixColumns = 9;
    nNewMatrixColumns = 7;
    % creating map
    pciMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:size(pciVocabulary, 1)
        pciKey = mat2str(pciVocabulary{i, 2});
        pciMap(pciKey) = pciVocabulary{i, 1};
    end
    
    newDataSet = cell(size(dataSet));
    
    for i = 1:size(dataSet, 1)     
        newDataSet{i, 1} = dataSet{i, 1};
        newDataSet{i, 2} = dataSet{i, 2};
        newDataSet{i, 4} = dataSet{i, 4};
        pciValues = dataSet{i, 3};
        newMatrix = zeros(size(pciValues, 1), nNewMatrixColumns);
        newMatrix (:,2:nNewMatrixColumns) = pciValues(:,  [3,4,5,6,7,9]);
    
        for j = 1:size(pciValues, 1)
            pciKey = mat2str(pciValues(j, [1 2 8]));
            newMatrix(j, 1) = pciMap(pciKey); % assign ID
        end
        newDataSet{i, 3} = newMatrix;
    end

end
function [newDataSet, pciVocabulary] = PCIstring2ID(dataSet)

    % to check if this works, use this, then PCID2string, then use
    % 'isequaln' to compare them and check if they are equal. 'isequal'
    % doesn't get along well with NaN's

    % structure: 3 numbers for uniquePCI_ID (columns 1, 2, 8), 4 Power Parameters, 1 Time parameter, 1 campaign ID.
    % final structure: 1 number for uniquePCI_ID.

    uniquePCIsList = cell2mat(dataSet(:, 3));
    uniquePCIsList = unique(uniquePCIsList(:, [1 2 8]), 'rows');
    nOriginalMatrixColumns = 9;
    nNewMatrixColumns = 7;

    
    pciVocabulary = cell(size(uniquePCIsList, 1), 2);
    for i = 1:size(uniquePCIsList, 1)
        pciVocabulary{i, 1} = i; % this gets the ID
        pciVocabulary{i, 2} = uniquePCIsList(i, :); % % this grabs the combination array
    end
    
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
        newMatrix (:,2:nNewMatrixColumns) = pciValues(:, [3,4,5,6,7,9]);
    
        for j = 1:size(pciValues, 1)
            pciKey = mat2str(pciValues(j, [1 2 8]));
            newMatrix(j, 1) = pciMap(pciKey); % assign ID
        end
        newDataSet{i, 3} = newMatrix;
    end


end
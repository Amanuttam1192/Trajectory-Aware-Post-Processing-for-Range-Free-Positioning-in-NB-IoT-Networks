function [reconvDataSet] = PCIID2string (newDataSet, pciVocabulary)

    %In case one wants to use some scripts that rely on the 4x
    % double PCI identifier

    reconvDataSet = cell(size(newDataSet));
    for i = 1:size(newDataSet, 1)
        reconvDataSet{i, 1} = newDataSet{i, 1};
        reconvDataSet{i, 2} = newDataSet{i, 2};
        reconvDataSet{i, 4} = newDataSet{i, 4};
        pciValues = newDataSet{i, 3};
        newMatrix = zeros(size(pciValues, 1), 9);
        newMatrix (:,[3 4 5 6 7 9]) = pciValues(:, 2:7);
        for j = 1:size(pciValues, 1) 
            pciID = pciValues(j, 1);
            newMatrix(j,[1 2 8]) = getOriginalPCI(pciVocabulary, pciID); 
        end
    
        reconvDataSet{i, 3} = newMatrix; % stands for "reconverted Data Set"
   
    end
    
    % using vocabulary here
    function pciOriginal = getOriginalPCI(pciVocabulary, pciID)
        pciOriginal = pciVocabulary{pciID, 2};
    end

end
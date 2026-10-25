function filteredTestSet = filterUnknownPCIs(testSet, pciVocabulary)
    % Function to delete from the testSet entries related to uniqueNPCIs
    % not included in the testSet
    %
    % INPUTS:
    % - testSet: dataset to filter
    % - pciVocabulary: cell array containing known uniqueNPCIs
    %   (first column: numeric ID, second column: uniqueNPCI ID with three values: NPCI, eNB-ID, MNC)
    %
    % OUTPUT:
    % - filteredTestSet: filtered testSet without unknown uniqueNPCIS

    % Extract the three values associated to known uniqueNPCIs from pciVocabulary
    knownPCIs = cell2mat(pciVocabulary(:, 2)); % Convert the second column to a matrix
    removedLines = 0;
    
    % Initialize the filtered testSet
    filteredTestSet = testSet;

    % Iterate on each row in the testSet
    for i = size(testSet, 1):-1:1 
        % Extract current data (third column)
        payload = testSet{i, 3}; % Matrix associated to current entry
        
        % Check which rows in current data include unknown uniqueNPCIs
        isKnown = false(size(payload, 1), 1); 
        for j = 1:size(payload, 1)
            % Estract uniqueNPCI ID with three values
            pciIdentifier = payload(j, [1 2 8]); 
            
            % Check if the uniqueNPCIs is known
            isKnown(j) = any(ismember(knownPCIs, pciIdentifier, 'rows'));
        end
        
        % Filter data to keep only known uniqueNPCIs
        total = length(isKnown);
        removedLines = removedLines + total - sum(isKnown);
        payload = payload(isKnown, :);
        
        % If data is now empty, remove entire row
        if isempty(payload)
            filteredTestSet(i, :) = [];
        else
            % Add filtered data to the filtered testSet
            filteredTestSet{i, 3} = payload;
        end
    end
end

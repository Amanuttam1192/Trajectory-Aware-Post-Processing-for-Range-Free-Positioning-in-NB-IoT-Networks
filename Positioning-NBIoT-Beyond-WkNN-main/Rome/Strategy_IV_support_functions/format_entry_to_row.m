function formatted_row = format_entry_to_row(entry, unique_pcis, featureID)

    % feature ID: 2, 3, 4, 5
    % Inizialize the output vector to -inf
    num_pcis = length(unique_pcis);
    formatted_row = -inf(1, num_pcis);
    
    % Extract the PCI matrix  from the selected entry
    pci_matrix = entry{3};
    
    % Assign PCI feature value in the correct format
    for j = 1:size(pci_matrix, 1)
        pci_index = find(unique_pcis == pci_matrix(j, 1));
        formatted_row(pci_index) = pci_matrix(j, featureID); 
    end
end

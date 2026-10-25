function estimated_position = estimate_position_RF(membership_vector, centers)
    % Evaluate position as weighted sum of centroids
    membership_vector = normalize_rows(membership_vector);
    estimated_position = membership_vector * centers;
end


function normalized_matrix = normalize_rows(matrix)
    % Function to normalize rows in a matrix
    % so that the sum of each row is one
    %
    % Input:
    %   matrix - matrix to be normalized
    %
    % Output:
    %   normalized_matrix - normalized matrix 

    % Evaluate sum of each row
    row_sums = sum(matrix, 2);

    % Normalize each row dividing its elements by their sum
    normalized_matrix = bsxfun(@rdivide, matrix, row_sums);

    % % Deal with zero sum rows
    normalized_matrix(isnan(normalized_matrix)) = 0;
end

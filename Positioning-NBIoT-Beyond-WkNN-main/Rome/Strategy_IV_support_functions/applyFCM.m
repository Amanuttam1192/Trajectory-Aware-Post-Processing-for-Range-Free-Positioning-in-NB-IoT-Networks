function [X, Y, XVal, YVal, centers,dataSet_UTM] = applyFCM (dataSet, num_clusters, m, bounds, featureID, graphs, valPerc)

% parameters for FCM clustering
% m: fuzzification parameter
% num_clusters: number of clusters
%featureID: % 2 (RSSI), 3 (SINR), 4 (RSRP), 5 (RSRQ).

removeINF = false;
normalize = false;

% dataset formatting parameters
minRefValue = bounds(1); % reported value for missing PCI
maxRefValue = bounds(2); % maximum value for PCI, useful in thesis to remove outliers


% Convert to UTM 
dataSet_UTM = utmConversion(dataSet);





% Fuzzy C-Means
[centers, U] = fcm(cell2mat(dataSet_UTM(:, 1:2)), num_clusters, [m NaN NaN 0]);

% creating dataset
num_points = size(dataSet_UTM, 1);
unique_pcis = unique(cell2mat(cellfun(@(x) x(:, 1), dataSet_UTM(:, 3), 'UniformOutput', false)));

X = -inf(num_points, length(unique_pcis));

Y = zeros(num_points, num_clusters); % groundtruth (membership vector)


% assign values to dataset
for i = 1:num_points
    X(i, :) = format_entry_to_row(dataSet_UTM(i, :), unique_pcis, featureID); % flat array for given parameter
    Y(i, :) = U(:, i)'; % ground truth
end

if graphs
    values = X(:);
    figure
    histogram(values, 50);  
    xlabel('Range di Valori');
    ylabel('Frequenza');
    title('Distribution of non-infinite values of the chosen parameter');

end

% Create a mask to identify finite values (excluding -inf)
isFinite = isfinite(X);

% apply bounds
X(isFinite & X < minRefValue) = minRefValue;
X(isFinite & X > maxRefValue) = maxRefValue;


if graphs
    values = X(:);
    figure
    histogram(values, 50); 
    xlabel('Range');
    ylabel('Frequence');
    title('Distribution of non-infinite values after applying bounds');
end


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

if removeINF && normalize && graphs
    values = X(:);
    figure
    histogram(values, 50);  
    xlabel('Range');
    ylabel('Frequence');
    title('Distribution of values after removing INFs and normalizing');

end

% further NaN management

rows_with_nan = any(isnan(X), 2);
X = X(~rows_with_nan, :);
Y = Y(~rows_with_nan, :);

numRows = size(X, 1);
numSampleRows = round(valPerc * numRows/100);

% Generate random indexex for 5% of the rows
randomIndices = randperm(numRows, numSampleRows);

% Create XVal e YVal with random samples
XVal = X(randomIndices, :);
YVal = Y(randomIndices, :);

% Remove these rows from X and Y to build the the final sets
X(randomIndices, :) = [];
Y(randomIndices, :) = [];






%coordinates_train = coordinates_train(~rows_with_nan, :);





% IGNORARE, parte di salvataggio

% [fileName, filePath] = uiputfile('*.mat', 'Salva il file come');
% 
% if ischar(fileName) 
%     fullFileName = fullfile(filePath, fileName);
%     save(fullFileName, 'X', 'Y', 'centers', "coordinates_train");
% else
%     disp('saving canceled');
% end
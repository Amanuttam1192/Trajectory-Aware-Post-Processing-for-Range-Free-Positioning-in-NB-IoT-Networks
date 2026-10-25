function dataSet = utmConversion(dataSet)


 % Columns 1 and 2 from dataset: latitude and longitude
    lat = cell2mat(dataSet(:, 1));
    lon = cell2mat(dataSet(:, 2));
    
    % Determine the UTM zone for all coordinates
    zoneNumbers = floor((lon + 180) / 6) + 1; % Commpute the UTM zone
    isNorthern = lat >= 0; % Determine the hemisphere (true for North, false for South)
    
    % Compute the EPSG code for each point
    epsgCodes = 32600 + zoneNumbers; % EPSG base for North hemisphere
    epsgCodes(~isNorthern) = 32700 + zoneNumbers(~isNorthern); % Adjustment for South hemisphere
    
    % Conversion using projfwd
    utm_x = zeros(size(lat));
    utm_y = zeros(size(lon));
    
    for i = 1:length(lat)
        % Create the reference system using the EPSG code
        proj = projcrs(epsgCodes(i)); % EPSG code for each point
        
        % Convert lat/lon -> UTM
        [utm_x(i), utm_y(i)] = projfwd(proj, lat(i), lon(i));
    end
    
    % Replace coordinates in dataset
    dataSet(:, 1) = num2cell(utm_x); % Column 1: UTM X (Easting)
    dataSet(:, 2) = num2cell(utm_y); % Column 2: UTM Y (Northing)

end
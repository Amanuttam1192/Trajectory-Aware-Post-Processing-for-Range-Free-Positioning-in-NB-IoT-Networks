%This script implements Strategy V for the Rome dataset as described in the following research paper:
% L. De Nardis, M. Savelli, G. Caso, F. Ferretti, L. Tonelli, N. Bouzar, A. Brunstrom, Ö. Alay, M. Neri, F. Elbahhar and M.-G. Di Benedetto,
% “Range-free Positioning in NB-IoT Networks by Machine Learning: beyond WkNN”,
% IEEE Journal on Indoor and Seamless Positioning and Navigation, 2025.


clear
close all

%% Parameters for the experiments
% PCA configuration: 0 = no PCA, 1 = apply PCA.
PCA_vector=[95 100];
PCA_values = PCA_vector<100;

% Validation percentages (the fraction of the 70% training+validation set to use as validation).
Validation_values = [0.1];

% Validation frequency values (number of iterations between validations during training).
vf_values = [20];
nRuns = 2;  % Number of runs

%% Load the dataset
load('Rome_NBIoT_data.mat')
%load('Campaign_data_NBIoT_1_2_3_4_5_6_interp_Rome.mat')

%% Build the Feature Matrix (M)
% (This block builds M based on RF parameters.)
RSSI = 3;
SINR = 4;
RSRP = 5;
RSRQ = 6;
M = [];
RF_param = 4;  % Modify as needed
results=[];
switch RF_param
    case RSSI
        missRefValue = -160;
        paramStr='RSSI';
    case SINR
        missRefValue = -25;
        paramStr='SINR';
    case RSRP
        missRefValue = -160;
        paramStr='RSRP';
    case RSRQ
        missRefValue = -40;
        paramStr='RSRQ';
end

%Build of feature matrix for PCA input
% Extract the columns [1 2 8] from each cell in dataSet(:,3)
temp_data = cellfun(@(a) {a(:, [1 2 8])}, dataSet(:,3));
% Convert the cell array to a matrix to find unique NPCI IDs
NPCI_ID = cell2mat(cellfun(@(x) x, temp_data, 'UniformOutput', false));
uniqueNPCIs = unique(NPCI_ID, 'rows');

% Create an M_part filled with the missing reference value.
M= missRefValue * ones(size(dataSet,1), size(uniqueNPCIs,1));
for i = 1:size(dataSet,1)
    RFP_mat = cell2mat(dataSet(i,3));
    [~, lib] = ismember(RFP_mat(:,[1 2 8]), uniqueNPCIs, 'rows');
    valid_idx = find(~isnan(lib) & lib ~= 0);
    if ~isempty(valid_idx)
        M(i, lib(valid_idx)) = RFP_mat(valid_idx, RF_param);
    end
end

% Save the original feature matrix so that it can be re-used across different configurations.
M_orig = M;

%% Loop over all configurations
for npca = 1:length(PCA_values)
    currentPCA = PCA_values(npca);

    for val_idx = 1:length(Validation_values)
        currentValidation = Validation_values(val_idx);

        for vf_idx = 1:length(vf_values)
            currentVF = vf_values(vf_idx);

            %% Feature Matrix Processing: PCA vs. No PCA
            PCA_var_perc = PCA_vector(npca); % Percentage of variance to preserve
            if currentPCA
                PCA_time_timer=tic;
                for i=1:nRuns
                    [coeff, score, ~, ~, explained] = pca(M_orig);
                    sum_Impact = explained(1);
                    PCn = 1;
                    while sum_Impact < PCA_var_perc && PCn < length(explained)
                        PCn = PCn + 1;
                        sum_Impact = sum_Impact + explained(PCn);
                    end
                    M = score(:, 1:PCn);
                end
                PCA_time=toc(PCA_time_timer);
            else
                M=M_orig;
            end


            %% Dataset Partitioning and Preparation
            estimationErrorVector = [];
            estimationError = [];
            training_phase_times = zeros(1, nRuns);
            online_phase_times = zeros(1, nRuns);
            Variance = zeros(1, nRuns);

            for r = 1:nRuns
                % Convert GPS coordinates to UTM and normalize them.
                Points_coords = cell2mat(dataSet(:,1:2));
                [Points_UTM_x, Points_UTM_y, zone] = deg2utm(Points_coords(:,1), Points_coords(:,2));
                [Points_UTM_x_norm, min_lat, max_lat] = minmax_n(Points_UTM_x);  % Min-max normalization
                [Points_UTM_y_norm, min_lon, max_lon] = minmax_n(Points_UTM_y);  % Min-max normalization
                Points_coords_n = [Points_UTM_x_norm, Points_UTM_y_norm];         % Normalized coordinates

                % Partition the data using a holdout method (70% training+validation, 30% test)
                rng(r);  % For reproducibility
                c = cvpartition(size(dataSet, 1), 'Holdout', 0.3);
                % Always use the entire training portion (70%) for network training.
                trainIndices = find(training(c));  % 70% of the data
                testIndices  = find(test(c));        % 30% of the data

                % Create a validation set as a random subset of the training data,
                % but DO NOT remove these samples from training.
                if currentValidation > 0
                    numValidation = round(length(trainIndices) * currentValidation);
                    valIndices = trainIndices(randperm(length(trainIndices), numValidation));
                else
                    valIndices = [];
                end

                % Split the feature data.
                dataPredTrain = M(trainIndices, :);
                dataPredTest  = M(testIndices, :);
                if ~isempty(valIndices)
                    dataPredValidation = M(valIndices, :);
                end

                % Get the corresponding target coordinates (normalized).
                YTrain = Points_coords_n(trainIndices, :);
                YTest  = Points_coords_n(testIndices, :);
                if ~isempty(valIndices)
                    YValidation = Points_coords_n(valIndices, :);
                end

                % Also store the original coordinates for error analysis.
                testPointCoordinates = Points_coords(testIndices, :);
                trainingPointCoordinates = Points_coords(trainIndices, :);

                %% DLNN Model Definition
                numFeatures = size(M, 2);
                layers = [
                    featureInputLayer(numFeatures, 'Normalization', 'rescale-zero-one')
                    fullyConnectedLayer(300)
                    batchNormalizationLayer
                    reluLayer
                    fullyConnectedLayer(100)
                    batchNormalizationLayer
                    reluLayer
                    fullyConnectedLayer(2)
                    regressionLayer];

                miniBatchSize = 48;

                % Define training options.
                if ~isempty(valIndices)
                    options = trainingOptions('adam', ...
                        'MiniBatchSize', miniBatchSize, ...
                        'Shuffle', 'every-epoch', ...
                        'MaxEpochs', 80, ...
                        'InitialLearnRate', 0.1, ...
                        'LearnRateDropPeriod', 20, ...
                        'LearnRateSchedule', 'piecewise', ...
                        'Verbose', true, ...
                        'ValidationData', {dataPredValidation, YValidation}, ...
                        'ValidationFrequency', currentVF);
                else
                    options = trainingOptions('adam', ...
                        'MiniBatchSize', miniBatchSize, ...
                        'Shuffle', 'every-epoch', ...
                        'MaxEpochs', 80, ...
                        'InitialLearnRate', 0.01, ...
                        'LearnRateDropPeriod', 20, ...
                        'LearnRateSchedule', 'piecewise', ...
                        'Verbose', true);
                end

                %% Training the Neural Network
                training_phase_tmp_r = tic;
                net = trainNetwork(dataPredTrain, YTrain, layers, options);
                training_phase_times(r) = toc(training_phase_tmp_r);

                %% Prediction Phase (Testing)
                online_phase_tmp_r = tic;
                estimatedTestPointCoordinates = predict(net, dataPredTest);
                online_phase_times(r) = toc(online_phase_tmp_r);

                % Convert predicted normalized coordinates back to the original values.
                estimatedTestPointCoordinates(:,1) = inv_minmax(estimatedTestPointCoordinates(:,1), min_lat, max_lat);
                estimatedTestPointCoordinates(:,2) = inv_minmax(estimatedTestPointCoordinates(:,2), min_lon, max_lon);
                [estimatedTestPointCoordinates(:,1), estimatedTestPointCoordinates(:,2)] = ...
                    utm2deg(estimatedTestPointCoordinates(:,1), estimatedTestPointCoordinates(:,2), zone(1:size(estimatedTestPointCoordinates,1), :));

                % Compute estimation error (in meters)
                tmp = 1000 * haversine_array(testPointCoordinates, ...
                    estimatedTestPointCoordinates(:,1), estimatedTestPointCoordinates(:,2));

                Variance(r) = var(tmp);
                stdDev(r)=std(tmp,1);
                estimationErrorVector = [estimationErrorVector; tmp];
                estimationError = [estimationError, mean(tmp)];

                close all force
            end

            %% Performance Analysis and Saving Results
            err_mean = mean(estimationError);
            err_variance = mean(Variance);
            err_std=mean(stdDev);
            err_mean_std=std(estimationError);

            % Build an informative filename.
            %filename = sprintf('Rome_0.1_ANN_PCA_%d_Val_%.2f_vf_%d_Rome_Opti.mat', currentPCA, currentValidation, currentVF);
            filename = sprintf('Strategy_V_Rome_RF_param_%s_PCA_%d_Val_%.2f_vf_%d.mat', paramStr, PCA_var_perc, currentValidation, currentVF);
            % Save the entire workspace.
            save(filename);

            fprintf('Saved all workspace for configuration: PCA=%d, Validation=%.2f, VF=%d\n', PCA_var_perc, currentValidation, currentVF);
            Training_time=sum(training_phase_times);
            Offline_time = Training_time+PCA_time;
            Online_time  = sum(online_phase_times);
            OffTimePCA_V_Rome(npca) = PCA_time/nRuns;
            OffTime_V_Rome(npca) = Offline_time/nRuns;
            OnTime_V_Rome(npca) = Online_time/nRuns;
            errAverage_V_Rome{npca}=err_mean;
            errVariance_V_Rome{npca}=err_variance;
            errStd_V_Rome{npca}=err_std;
            results = [results; PCA_var_perc, err_mean, err_std, err_variance, err_mean_std, PCA_time Training_time Offline_time Online_time];
        end
    end
end
save PCA_Sweep_Strategy_V_Rome errAverage_V_Rome errVariance_V_Rome errStd_V_Rome OffTimePCA_V_Rome OffTime_V_Rome OnTime_V_Rome nRuns PCA_vector results
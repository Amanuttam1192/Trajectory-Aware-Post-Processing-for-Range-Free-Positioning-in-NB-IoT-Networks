% This script implements Strategy I for the Oslo dataset as described in the following research paper:
% L. De Nardis, M. Savelli, G. Caso, F. Ferretti, L. Tonelli, N. Bouzar, A. Brunstrom, Ö. Alay, M. Neri, F. Elbahhar and M.-G. Di Benedetto, 
% “Range-free Positioning in NB-IoT Networks by Machine Learning: beyond WkNN”, IEEE Journal on Indoor and Seamless Positioning and Navigation, 2025.
clc
clear
close all;

%Data import
dataFileName=sprintf('Oslo_NBIoT_data.mat');
fExists=exist(dataFileName, 'file');
if(fExists==2)
    load(dataFileName);
else
    exit
end

% Inizialization of the Test Points / Reference Points split
TP_RP_preallocation_flag=1;
if TP_RP_preallocation_flag
    load 'TP_RP_allocation_Oslo';
end

%Indexes of radio features in the dataset entries: [3 = RSSI, 4 = SINR, 5 = RSRP, 6 = RSRQ]
RSSI=3;
SINR=4;
RSRP=5;
RSRQ=6;

RF_param= SINR; %Selector for RF parameter

switch RF_param %Missing feature value settings
    case RSSI
        missRefValue=-160;
        paramStr='RSSI';
    case SINR
        missRefValue=-25;
        paramStr='SINR';
    case RSRP
        missRefValue=-160;
        paramStr='RSRP';
    case RSRQ
        missRefValue=-40;
        paramStr='RSRQ';
end

%Build of feature matrix for PCA input
% Extract the columns [1 2 8] from each cell in dataSet(:,3)
temp_data = cellfun(@(a) {a(:, [1 2 8])}, dataSet(:,3));
% Convert the cell array to a matrix to find unique NPCI IDs
NPCI_ID = cell2mat(cellfun(@(x) x, temp_data, 'UniformOutput', false));
uniqueNPCIs = unique(NPCI_ID, 'rows');

% Create an M matrix filled with the missing reference value.

M = missRefValue * ones(size(dataSet,1), size(uniqueNPCIs,1));
for i = 1:size(dataSet,1)
    RP_mat = cell2mat(dataSet(i,3));
    [~, lib] = ismember(RP_mat(:,[1 2 8]), uniqueNPCIs, 'rows');
    valid_idx = find(~isnan(lib) & lib ~= 0);
    if ~isempty(valid_idx)
        M(i, lib(valid_idx)) = RP_mat(valid_idx, RF_param);
    end
end
% Save the original feature matrix so that it can be re-used across different configurations.
M_orig = M;

kMax=4; %Maximum value of k for WkNN
nRuns = 100; % number of runs to be used
PCA = 1; %PCA flag
%PCA_var_percVect=70:5:100; %Values for sweep over PCA percentage of variance
PCA_var_percVect=[95,100]; %Values for sweep over PCA percentage of variance
results = [];

for npca=1:length(PCA_var_percVect)
    PCA_var_perc=PCA_var_percVect(npca);
    fprintf('PCA percentage: %d\n', PCA_var_perc);
    time=tic;
    % PCA analysis (repeated nRuns times to evaluate the average required time)

    if (PCA == 1) & (PCA_var_perc <100)
        for q=1:nRuns
            [~,score,~,~,explained] = pca(M_orig);
            sum_Impact = explained(1);
            PCn = 1;
            while sum_Impact<PCA_var_perc
                PCn = PCn+1;
                sum_Impact = sum_Impact + explained(PCn);
            end

            M = score(:,1:PCn);
        end
        PCA_flag=1;
    else
        M=[];
        PCA_flag=0;
    end
    PCA_time = toc(time);
    Previous = PCA_time;
    %End of computing time evaluation for PCA

    [tp_errors] = deal(cell(length(PCA_var_percVect), nRuns));
    percLocated=zeros(length(PCA_var_percVect),nRuns);
    
    %WkNN positioning
    for k = 1:nRuns
        if TP_RP_preallocation_flag
            allocationVector=TP_RP_allocation(:,k);
        else
            allocationVector=[];
        end

        
        [tp_errors{k},percLocated(k),TPs_located, best_k, W]=Wei_Cov_strategy(dataSet, RF_param, kMax,allocationVector,PCA_flag,M,missRefValue);
        err_mean_run(k,:)=mean(tp_errors{k}.*1000);% error mean (in m) across Test Points observed in each run
        err_variance_run(k,:) = var(tp_errors{k}.*1000); % error variance (in m^2) across Test Points observed in each run
        err_std_run(k,:) = std(tp_errors{k}.*1000); % error standard deviation (in m) across Test Points observed in each run
        if k == 1
            tp_errors_total = tp_errors{1};
        else
            tp_errors_total = [tp_errors_total ; tp_errors{k}]; % cumulative error for all Test Points in all runs
        end
    end
    Positioning_time = toc(time) - Previous;
    %End of computing time evaluation for WkNN

    err_variance = mean(err_variance_run); % error variance (in m^2) across Test Points averaged over the runs
    err_std = mean(err_std_run); % error standard deviation (in m) across Test Points averaged over the runs
    err_mean = mean(err_mean_run); % average error (in m) across Test Points averaged over the runs
    err_mean_std = std(err_mean_run); % standard deviation (in m) of average error across Test Points

    save(['Strategy_I_Oslo_RF_param_' paramStr '_PCA_Perc_' num2str(PCA_var_perc) '_' num2str(nRuns) '_runs.mat']);
    %The following instructions save for each PCA percentage the values used for the performance analysis in the
    %reference research paper
    results = [results; PCA_var_percVect(npca), err_mean, err_std, err_variance err_mean_std PCA_time Positioning_time];
    OnTime_I_Oslo(npca)=(Positioning_time)/nRuns;
    OffTime_I_Oslo(npca)= PCA_time/nRuns;
    errAverage_I_Oslo{npca}=err_mean;
    errVariance_I_Oslo{npca}=err_variance;
    errStd_I_Oslo{npca}=err_std;
end

save PCA_Sweep_Strategy_I_Oslo errAverage_I_Oslo errVariance_I_Oslo errStd_I_Oslo OffTime_I_Oslo OnTime_I_Oslo nRuns PCA_var_percVect results

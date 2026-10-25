%This script implements Strategy IV for the Oslo dataset as described in the following research paper:
% L. De Nardis, M. Savelli, G. Caso, F. Ferretti, L. Tonelli, N. Bouzar, A. Brunstrom, Ö. Alay, M. Neri, F. Elbahhar and M.-G. Di Benedetto,
% “Range-free Positioning in NB-IoT Networks by Machine Learning: beyond WkNN”, IEEE Journal on Indoor and Seamless Positioning and Navigation, 2025.

clear
clc
close all

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

%Indexes of radio features in the dataset entries
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
nRuns = 2; % number of runs to be used
R_fp = 700; % radius in meters of area centered around the coarse position determining the subset used during the Fine positioning step
%NclustVect = [2,3,4,5,10,15,20,25]; %Values for sweep over number of clusters
%NclustVect = [5,10,15,20,25]; %Values for sweep over number of clusters
NclustVect = 10; %Values for sweep over number of clusters

PCA = 1; %PCA flag
%PCA_var_percVect=95:5:100; %Values for sweep over PCA percentage of variance
PCA_var_percVect=[95,100];

Num_of_RP = length(find(TP_RP_allocation(:,1) == 1)); % number of RPs
Num_of_TP = length(find(TP_RP_allocation(:,1) == 2)); % number of TPs
%Points_coords = cell2mat(dataSet(:,1:2)); % Coordinates of all points

TrainingPerc = 70; % Data percentage used as Training Set

Mvect = [1.3];
results = [];
dataSetOrig=dataSet;
for npca=1:length(PCA_var_percVect)
    PCA_var_perc=PCA_var_percVect(npca);
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
            dataSet=rebuildDataSetFromFeatures(dataSetOrig,M,RF_param);
        end
    else
        dataSet=dataSetOrig;
        M=M_orig;        
    end
    PCA_time = toc(time);
    Previous = PCA_time;
    for ii = 1:length(NclustVect)
        for jj = 1:length(Mvect)
            fprintf('PCA percentage: %d; Number of clusters: %d; m: %f\n', PCA_var_perc, NclustVect(ii), Mvect(jj));
            %if PCA==0
            %    maxRefValue=40;
            %else
            maxRefValue=max(max(M));
            %end

            bounds = [-inf maxRefValue];

            %Offline phase: Clustering and training of RF models
            for r = 1:nRuns
                [centers{r},trainingSet{r}, testSet{r}, models{r}, XTest{r},trainingSet_UTM{r}] = FCM_ranforest_alloc_vec(dataSet, TrainingPerc, TP_RP_allocation(:,r), NclustVect(ii),Mvect(jj) , RF_param-1, bounds);
            end
            Clustering_training_time = toc(time)-Previous;
            Previous = Previous+Clustering_training_time;

            %Online phase: Coarse and fine estimation

            %Coarse estimation
            for r = 1:nRuns
                coarseCoordsTest{r} = coarsePosition(XTest{r}, models{r}, centers{r});
            end
            Coarse_estimation_time = toc(time) - Previous;
            Previous = Previous + Coarse_estimation_time;
            
            %Extraction of relevant subset based on coarse estimation
            percLocated=zeros(1,nRuns);
            for k = 1:nRuns
                subSetIndex{k}  = getSubsetIDX(trainingSet_UTM{k}, coarseCoordsTest{k}, R_fp);
            end
            Subset_extraction_time = toc(time) - Previous;
            Previous = Previous + Subset_extraction_time;

            %Fine estimation
            for k = 1:nRuns
                if TP_RP_preallocation_flag
                    allocationVector=TP_RP_allocation(:,k);
                else
                    allocationVector=[];
                end

                [tp_errors{k},percLocated(1,k),TPs_located, best_k, W]=Wei_Cov_strategy_FCM(trainingSet{k},testSet{k},subSetIndex{k}, RF_param,kMax, missRefValue);
                err_mean_run(k,:)=mean(tp_errors{k}.*1000);% error mean (in m) across Test Points observed in each run
                err_variance_run(k,:) = var(tp_errors{k}.*1000); % error variance (in m^2) across Test Points observed in each run
                err_std_run(k,:) = std(tp_errors{k}.*1000); % error standard deviation (in m) across Test Points observed in each run
                if k == 1
                    tp_errors_total = tp_errors{1};
                else
                    tp_errors_total = [tp_errors_total ; tp_errors{k}]; % matrice di accumulo di tutti gli errori in tutte le run
                end
            end
            Fine_estimation_time = toc(time) - Previous;
            %End of computing time evaluation for WkNN


            err_variance = mean(err_variance_run,1); % error variance (in m^2) across Test Points averaged over the runs
            err_std = mean(err_std_run,1); % error standard deviation (in m) across Test Points averaged over the runs
            err_mean = mean(err_mean_run,1); % average error (in m) across Test Points averaged over the runs
            err_mean_std = std(err_mean_run,1,1); % standard deviation (in m) of average error across Test Points
            save(['Strategy_IV_Oslo_RF_param_' paramStr '_numClusters_' num2str(NclustVect(ii)) '_M_' num2str(Mvect(jj)) '_PCA_Perc_' num2str(PCA_var_perc) '_' num2str(nRuns) '_runs.mat']);

            %fprintf('num_clusters: %d, m: %.1f, average_error: %.2f\n', NclustVect(ii), Mvect(jj), min_average_error);
            Offline_time=PCA_time+Clustering_training_time;
            Online_time=Coarse_estimation_time+Subset_extraction_time+Fine_estimation_time;
            % Store results in the structure
            results = [results; PCA_var_percVect(npca), NclustVect(ii), Mvect(jj), err_mean, err_std, err_variance, err_mean_std, PCA_time Clustering_training_time Offline_time Coarse_estimation_time Subset_extraction_time Fine_estimation_time Online_time];
            OnTime_IV_Oslo(ii,npca)=(Online_time)/nRuns;
            OffTime_IV_Oslo(ii,npca)=(Offline_time)/nRuns;
            errAverage_IV_Oslo{ii,npca}=err_mean;
            errVariance_IV_Oslo{ii,npca}=err_variance;
            errStd_IV_Oslo{ii,npca}=err_std;
        end
    end
end
save K_m_Sweep_Strategy_IV_Oslo errAverage_IV_Oslo errVariance_IV_Oslo OffTime_IV_Oslo OnTime_IV_Oslo nRuns Mvect NclustVect results


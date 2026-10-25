% This script implements Strategy III for the Oslo dataset as described in the following research paper:
% L. De Nardis, M. Savelli, G. Caso, F. Ferretti, L. Tonelli, N. Bouzar, A. Brunstrom, Ö. Alay, M. Neri, F. Elbahhar and M.-G. Di Benedetto,
% “Range-free Positioning in NB-IoT Networks by Machine Learning: beyond WkNN”, IEEE Journal on Indoor and Seamless Positioning and Navigation, 2025.

%Data import
dataFileName=sprintf('Rome_NBIoT_data.mat');
fExists=exist(dataFileName, 'file');
if(fExists==2)
    load(dataFileName);
else
    exit
end

% Inizialization of the Test Points / Reference Points split
TP_RP_preallocation_flag=1;
if TP_RP_preallocation_flag
    load 'TP_RP_allocation_Rome';
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
M(isnan(M))=missRefValue;
% Save the original feature matrix so that it can be re-used across different configurations.
M_orig = M;

kMax=4; %Maximum value of k for WkNN
nRuns = 100; % number of runs to be used
Clust_strat = 2; % clustering strategy (1=kmedoids 2=kmeans)
%NclustVect = [2,3,4,5,10,15,20,25]; %Values for sweep over number of clusters
%NclustVect = [5,10,15,20,25]; %Values for sweep over number of clusters
NclustVect = 10; %Values for sweep over number of clusters

PCA = 1; %PCA flag
%PCA_var_percVect=75:5:100; %Values for sweep over PCA percentage of variance
PCA_var_percVect=[95,100];

Num_of_RP = length(find(TP_RP_allocation_Rome(:,1) == 1)); % number of RPs
Num_of_TP = length(find(TP_RP_allocation_Rome(:,1) == 2)); % number of TPs
Points_coords = cell2mat(dataSet(:,1:2)); % Coordinates of all points
results = [];
for nc=1:length(NclustVect)
    Nclust=NclustVect(nc); % Number of clusters used to split the RPs
    for npca=1:length(PCA_var_percVect)
        PCA_var_perc=PCA_var_percVect(npca);
        fprintf('Number of clusters: %d; PCA percentage: %d\n', Nclust, PCA_var_perc);
        time = tic;

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
            M=M_orig;
            PCA_flag=0;
        end
        PCA_time = toc(time);
        Previous = PCA_time;
        %End of PCA analysis

        %Clustering
        RP = zeros(Num_of_RP,nRuns);
        TP = zeros(Num_of_TP,nRuns);
        clusters = zeros(Num_of_RP,nRuns);

        for r = 1:nRuns
            mRP = find(TP_RP_allocation_Rome(:,r) == 1);
            include = sort(randperm(length(mRP),round(length(mRP))));
            RP(:,r) = mRP(include);
            RP_coords(:,2*r-1:2*r) = Points_coords(RP(:,r),:);
            RP_coords_tmp = RP_coords(:,2*r-1:2*r);

            TP(:,r) = find(TP_RP_allocation_Rome(:,r) == 2);
            TP_coords(:,2*r-1:2*r) = Points_coords(TP(:,r),:);
            TP_coords_tmp = TP_coords(:,2*r-1:2*r);
            switch Clust_strat
                case 1
                    [clusters(:,r),Centroids] = kmedoids(RP_coords_tmp,Nclust,'OnlinePhase','on');
                case 2
                    [clusters(:,r),Centroids] = kmeans(RP_coords_tmp,Nclust,'OnlinePhase','on');
            end

        end

        Clustering_time = toc(time)-Previous;
        Previous = Previous + Clustering_time;
        % End of clustering

        %RF Classifier training and inference
        predLabels = zeros(Num_of_TP,nRuns);


        % Training
        for r = 1:nRuns
            Train_Features = M(RP(:,r),:);
            Test_Features{r} = M(TP(:,r),:);
            Train_Labels = clusters(:,r);

            Classes = 1:Nclust;

            % Random Forest classifiers
            RF_classifier{r} = fitcensemble(Train_Features,Train_Labels,'Method','Bag','Classnames',Classes);

        end

        Training_time = toc(time) - Previous;
        Previous = Previous + Training_time;
        %End of RF training

        %Testing
        for r = 1:nRuns
            predLabels(:,r) = predict(RF_classifier{r},Test_Features{r});
        end

        Testing_time = toc(time) - Previous;
        Previous = Previous + Testing_time;
        %End of RF testing

        %WkNN positioning

        percLocated=zeros(1,nRuns);

        for k = 1:nRuns
            shift = 0; % support variable to create overall data vector
            if TP_RP_preallocation_flag
                allocationVector=TP_RP_allocation_Rome(:,k);
            else
                allocationVector=[];
            end

            for c = 1:Nclust  
                RP_in_clust_c=RP(clusters(:,k)==c,k); % indexes of RPs in cluster c
                TP_in_clust_c = TP(predLabels(:,k)==c,k); % indexes of TPs assigned to cluster c by the classifier
                nTPSC(c)=length(TP_in_clust_c);
                if ~isempty(TP(predLabels(:,k)==c,k)) % check whether there is at least a TP assigned to cluster c

                    indexes_vector = [RP_in_clust_c ; TP_in_clust_c]; % indexes of points selected in cluster c
                    Points_in_clust = sort(indexes_vector); % sorted indexes
                    subSet = dataSet(Points_in_clust,:); % select points in cluster c from the dataset
                    clust_alloc_vector = TP_RP_allocation_Rome(Points_in_clust,k);  % allocation vector for cluster c
                    M_sub = M(Points_in_clust,:);
                    [tp_errors_cluster{c,k},percLocated(k),TPs_located, best_k, W]=Wei_Cov_strategy(subSet, RF_param,kMax,clust_alloc_vector,PCA_flag,M_sub,missRefValue);
                    average_error_cluster{c,k}=mean(tp_errors_cluster{c,k},1); % error mean (in m) across Test Points in each cluster
                    if tp_errors_cluster{c,k} ~= zeros(1,kMax) % check whether there is at least a TP in the cluster with an estimated position
                        if shift == 0
                            tp_errors{k} = tp_errors_cluster{c,k};
                            shift = 1;
                            average_error{k} = average_error_cluster{c,k};
                        else
                            tp_errors{k} = [tp_errors{k} ; tp_errors_cluster{c,k}];
                            average_error{k} = [average_error{k} ; average_error_cluster{c,k}];
                        end
                    else
                        fprintf('No TPs located in cluster %d, run %d; number of TPs assigned to the cluster: %d\n',c,k,nTPSC(c));
                    end
                end
            end

            err_mean_over_clusters_run(k,:) = mean(average_error{k}.*1000);  % error mean (in m) across clusters in each run
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
        %End of WkNN positioning


        err_variance = mean(err_variance_run,1); % error variance (in m^2) across Test Points averaged over the runs
        err_std = mean(err_std_run,1); % error standard deviation (in m) across Test Points averaged over the runs
        err_mean = mean(err_mean_run,1); % average error (in m) across Test Points averaged over the runs
        err_mean_over_clusters=mean(err_mean_over_clusters_run,1); % average error (in m) across clusters averaged over the runs
        err_mean_std = std(err_mean_run,1,1); % standard deviation (in m) of average error across Test Points
        save(['Strategy_III_RF_param_' paramStr '_numClusters_' num2str(Nclust) '_PCA_Perc_' num2str(PCA_var_perc) '_' num2str(nRuns) '_runs.mat']);

        Online_time=Testing_time+Positioning_time;
        Offline_time=PCA_time+Clustering_time+Training_time;
        results = [results; PCA_var_percVect(npca), NclustVect(nc), err_mean, err_std, err_variance, err_mean_over_clusters, err_mean_std, PCA_time Clustering_time Training_time Offline_time Testing_time Positioning_time Online_time];
        OnTime_III_Oslo(nc,npca)=(Positioning_time+Testing_time)/nRuns;
        OffTime_III_Oslo(nc,npca)=(PCA_time+Clustering_time+Training_time)/nRuns;
        OffTimePCA_III_Oslo(nc,npca)=(PCA_time)/nRuns;
        errAverage_III_Oslo{nc,npca}=err_mean;
        errVariance_III_Oslo{nc,npca}=err_variance;
        errStd_III_Oslo{nc,npca}=err_std;
    end
end
save K_PCA_Sweep_Strategy_III_Oslo errAverage_III_Oslo errVariance_III_Oslo errStd_III_Oslo OffTimePCA_III_Oslo OffTime_III_Oslo OnTime_III_Oslo nRuns PCA_var_percVect NclustVect results

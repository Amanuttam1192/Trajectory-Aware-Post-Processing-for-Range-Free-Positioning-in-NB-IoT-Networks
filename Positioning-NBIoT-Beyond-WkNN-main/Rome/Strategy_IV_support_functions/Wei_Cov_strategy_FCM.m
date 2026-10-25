function [tp_errors, percLocated, TPs_located, selected_RP_idx, W] = Wei_Cov_strategy_FCM(struct_RP,struct_TP,subSetIndex, RF_param,kMax,missRefValue)
warning off

lTP =size(subSetIndex,2);
temp_RP = cellfun(@(a){{a(:,[1 2 8])}},struct_RP(:,3));
RP_NPCI_ID_Op=cell2mat(cellfun(@(x) cell2mat(x),temp_RP,'un',0));
temp_TP = cellfun(@(a){{a(:,[1 2 8])}},struct_TP(:,3));
if iscell(temp_TP) == 0
    d={};
    d{1}=temp_TP;
    temp_TP=d;
end
TP_NPCI_ID_Op=cell2mat(cellfun(@(x) cell2mat(x),temp_TP,'un',0));
uniqueNPCIs = unique([TP_NPCI_ID_Op; RP_NPCI_ID_Op], 'rows');

M_RP = (missRefValue)*ones(size(struct_RP,1), size(uniqueNPCIs,1)); 
M_TP = (missRefValue)*ones(size(struct_TP,1), size(uniqueNPCIs,1));  
idx2=zeros(size(struct_RP,1), size(uniqueNPCIs,1));
idx1=zeros(size(struct_TP,1), size(uniqueNPCIs,1));

for i = 1:size(struct_RP,1)
    RP_mat=cell2mat(struct_RP(i,3));
    [~,lib] = ismember(RP_mat(:,[1 2 8]), uniqueNPCIs, 'rows');
    M_RP(i,lib(logical(lib))) = RP_mat(logical(lib),RF_param);
    potentialNPCIs=lib(logical(lib));
    actualNPCIs=potentialNPCIs(~isnan(RP_mat(logical(lib),RF_param)));
    idx2(i,actualNPCIs) = 1;
end
for i = 1:size(struct_TP,1)
    TP_mat=cell2mat(struct_TP(i,3));
    [~,lib] = ismember(TP_mat(:,[1 2 8]), uniqueNPCIs, 'rows');
    M_TP(i,lib(logical(lib))) = TP_mat(logical(lib),RF_param);
    potentialNPCIs=lib(logical(lib));
    actualNPCIs=potentialNPCIs(~isnan(TP_mat(logical(lib),RF_param)));
    idx1(i,actualNPCIs) = 1;
end

dummyTPs=~any(idx1,2);
nDummyTPs=nnz(dummyTPs);
dummyRPs=~any(idx2,2);
validRPs=any(idx2,2);

vectOptimization=0;
if(vectOptimization)
    D = pdist2(M_TP, M_RP, 'euclidean');
else
    for i = 1:size(M_TP,1)
        for j = 1:size(M_RP,1)
            D(i,j)=0;
            for q=1:size(M_TP,2)
                D(i,j)=D(i,j)+(M_TP(i,q)-M_RP(j,q)).^2;
            end
            D(i,j)=sqrt(D(i,j));
        end
    end
end

%This strategy technically works even when there are no NPCIs in common
%between TP and any RP. But in this case the position will be determined
%picking the first k RPs, so pretty much randomly. Out of fairness with
%strategies where TPs without NPCIs in common with any RP are excluded, we
%do the same here.
TPs_located=zeros(1,size(struct_TP,1));

L = zeros(size(struct_TP,1), size(struct_RP,1));    %% matrix of BTS elements in common
for i = 1:size(struct_TP,1)
    match = bsxfun(@and, idx1(i,:), idx2);  % match between i-th TP e all RPs
    s = sum(match,2);                       % sum of the number of matches (positions in common)
    z = find(s==0);                         % find zeros (RPs with no matches with the TP)
    nz = find(s~=0);                        % find non-zero values (RPs with at least one match with the TP)
    
    L(i,:) = s;
    bin = unique(L(i,:));  %% list without repetition of number of matches between TP and all RPs
    if (length(bin)>1 || (bin~=0))
        TPs_located(i)=1;
    else
        %fprintf('TP %d not located\n',i);
    end
    for j = nz'
        D(i,j) = D(i,j)/(s(j));%For RPs that have at least one match with the TP we divide the Euclidean distance by the number of matches
    end
    for j = z'
        D(i,j) = realmax;%For RPs that have no match with the TP we set the distance to infinity
    end
end
D(:,dummyRPs)=realmax;
D(~subSetIndex')=realmax;
TPs_located=logical(TPs_located);

%Reference Points at distance 0 are set at a small distance (1/20 of the
%minimum D>0) to avoid a division by 0 when determining the corresponding weight
D(D==0) = min(D(D~=0))/20;

% We sort the RPs in order of increasing distance
[D_sort, idx_sort] = sort(D,2);
W = 1./D_sort;          % weights set as the inverse of distances


%We extract the position of TPs (ground truth) to compute the error;
real_lat=cell2mat(struct_TP(:,1));
real_long=cell2mat(struct_TP(:,2));
real_position=[real_lat real_long];

%----------------WKNN with selected RF parameter---------------------------------------------

k=1:1:kMax;
for i=1:length(k)
    this_k = k(i);
    
    %We select the first k(i) RPs
    selected_RP_idx = idx_sort(TPs_located,1:this_k);
    
    lat_k_RP_matrix = reshape(cell2mat(struct_RP(selected_RP_idx(:),1)), size(selected_RP_idx));
    long_k_RP_matrix = reshape(cell2mat(struct_RP(selected_RP_idx(:),2)), size(selected_RP_idx));
    
    % sum of weighted coordinates
    sum_lat=sum(lat_k_RP_matrix.*(W(TPs_located,1:this_k)),2);
    sum_long=sum(long_k_RP_matrix.*(W(TPs_located,1:this_k)),2);
    
    % normalization by the sum of the weights
    lat_k_TP = sum_lat./sum(W(TPs_located,1:this_k),2);
    long_k_TP = sum_long./sum(W(TPs_located,1:this_k),2);

    if any(TPs_located) % check whether at least one Test Point was positioned (at least one NPCI in common with at least one RP)
        [tp_errors(:,i),~,~]=haversine_array(real_position(TPs_located,:),lat_k_TP,long_k_TP);
    else
        tp_errors = zeros(1,kMax); % otherwise we set the output variable to 0 
    end
end

percLocated=length(TPs_located)/(lTP-nDummyTPs);%Percentage of Test Points for which a position could be estimated

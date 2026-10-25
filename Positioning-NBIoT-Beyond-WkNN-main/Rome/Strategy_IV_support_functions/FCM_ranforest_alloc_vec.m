function [centers,trainingSet, testSet, models, XTest,trainingSet_UTM] = FCM_ranforest_alloc_vec(dataSet, perc, allocationVector, num_clusters, m, featureID, bounds)



verbose = false;
if ~isempty(allocationVector)
    type=allocationVector;
    irfp(type==1)=1;
    itp(type==2)=1;


    idx_TP = find(itp);
    idx_RFP = find(irfp);
    lRFP = nnz(irfp);
    lTP = nnz(itp);
    trainingSet=dataSet(idx_RFP,:);
    testSet=dataSet(idx_TP,:);
else
    [trainingSet, testSet] = splitNBIoT(dataSet, perc);
end


[~, pciVocabularyFULL] = PCIstring2ID(dataSet);
[trainingSet, pciVocabulary] = PCIstring2ID(trainingSet);

if size(pciVocabulary,1) ~= size(pciVocabularyFULL)
    if verbose == true
        disp('Attention: not all uniqueNPCIs are present in the subset')
        fprintf('Total number of UniqueNPCIs: %d\n', size(pciVocabularyFULL, 1))
        fprintf('UniqueNPCIs in the subset: %d\n', size(pciVocabulary, 1))
    end
    testSet = filterUnknownPCIs(testSet, pciVocabulary);
end
%%


valPerc = 0;
FCM_graphs = false;

[XTrain, YTrain, ~, ~, centers,trainingSet_UTM] = applyFCM(trainingSet, num_clusters, m, bounds, featureID, FCM_graphs, valPerc);

if verbose == true
    fprintf('Training models...\n', num_clusters)
end
[models] = trainRanForestFCM(XTrain, YTrain);
if verbose == true
    disp('Training complete')
end


XTest = getXTest(trainingSet, testSet, featureID, bounds, pciVocabulary);
trainingSet = PCIID2string(trainingSet, pciVocabulary);



end
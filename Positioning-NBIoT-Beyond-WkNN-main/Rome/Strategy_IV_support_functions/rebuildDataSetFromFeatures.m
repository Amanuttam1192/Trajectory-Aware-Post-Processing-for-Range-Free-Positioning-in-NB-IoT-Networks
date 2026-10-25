function [rBDataSet]=rebuildDataSetFromFeatures(dataSet,fMatrix, RF_param)

rBDataSet = cell(size(dataSet));

for i = 1:size(dataSet, 1)
    rBDataSet{i, 1} = dataSet{i, 1};
    rBDataSet{i, 2} = dataSet{i, 2};
    rBDataSet{i, 4} = dataSet{i, 4};

    pciValues = dataSet{i, 3};
    newMatrix = zeros(size(fMatrix,2), size(pciValues, 2));
    newMatrix (:,1) = 1:size(fMatrix,2);
    newMatrix(:, RF_param)=fMatrix(i,:)';

    rBDataSet{i, 3} = newMatrix;
end
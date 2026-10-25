function X = inv_minmax(Normalized_values,min_X,max_X)
%% INPUT
% Normalized_values = normalized vector or matrix
% min_X = minimum value in X before normalization
% max_X = maximum value in X before normalization

%% OUTPUT
% X = un-normalized vector or matrix

X = (Normalized_values).*(max_X - min_X) + min_X;
end
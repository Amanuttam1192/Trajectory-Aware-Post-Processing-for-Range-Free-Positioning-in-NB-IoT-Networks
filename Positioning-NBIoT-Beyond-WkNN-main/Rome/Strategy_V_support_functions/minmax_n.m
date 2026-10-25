function [Normalized_values,min_X,max_X] = minmax_n(X)
%% INPUT
% X = vector or matrix to be normalized according to the MinMax technique

%% OUTPUT
% Normalized_values = normalized vector or matrix
% min_X = minimum value in X
% max_X = maximum value in X

min_X = min(min(X));
max_X = max(max(X));

Normalized_values = (X - min_X)./(max_X - min_X);
end
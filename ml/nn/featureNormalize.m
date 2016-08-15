function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

%               Note that X is a matrix where each column is a 
%               feature and each row is an example. we perform 
%		the normalization separately for 
%               each feature. 


mu = mean(X);
sigma = std(X);

for i = 1:size(X,2)
	X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
end

% ============================================================

end

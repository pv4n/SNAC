function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	A = X * theta - y;  % (m x 1 vector)
	delta = 1 / m * (A' * X)';  % ' ((n+1) x 1 vector)
	theta = theta - (alpha * delta); % ' ((n+1) x 1 vector)

    % ============================================================

    % Save the cost J in every iteration    
	cost = computeCostMulti(X, y, theta);
	J_history(iter) = cost;

end

end

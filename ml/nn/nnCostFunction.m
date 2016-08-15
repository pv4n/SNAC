function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   %nnot mine
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%	Feedforward the neural network and return the cost in the
%       variable J. then verify the cost function computation is correct by verifying the cost
%       computed in ex4.m

K = num_labels;
X = [ones(m,1) X];

for i = 1:m
	X_i = X(i,:);
	h_of_Xi = sigmoid( [1 sigmoid(X_i * Theta1')] * Theta2' );
	
	% if y = 5 then y_i = [0 0 0 0 1 0 0 0 0 0]
	y_i = zeros(1,K);
	y_i(y(i)) = 1;
	
	J = J + sum( -1 * y_i .* log(h_of_Xi) - (1 - y_i) .* log(1 - h_of_Xi) );
end;

J = 1 / m * J;

% Add regularization term

J = J + (lambda / (2 * m) * (sum(sumsq(Theta1(:,2:input_layer_size+1))) + sum(sumsq(Theta2(:,2:hidden_layer_size+1))))); 

%
% 	  Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. we can check
%         implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. we map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

for t = 1:m
	a_1 = X(t,:);  
	z_2 = a_1 * Theta1';
	a_2 = [1 sigmoid(z_2)];
	z_3 = a_2 * Theta2';
	a_3 = sigmoid(z_3);
	y_i = zeros(1,K);
	y_i(y(t)) = 1;
	
	delta_3 = a_3 - y_i;
	delta_2 = delta_3 * Theta2 .* sigmoidGradient([1 z_2]);
	
	delta_accum_1 = delta_accum_1 + delta_2(2:end)' * a_1;
	delta_accum_2 = delta_accum_2 + delta_3' * a_2;
end;

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;

%	  Implement regularization with the cost function and gradients.

Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

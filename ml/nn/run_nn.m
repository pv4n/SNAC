#! /usr/bin/octave -qf

% depoyable nn by passing in command line args in the following manner:
% run_nn {flow} {bbox height} {center x} {center y} {depth}

input_layer_size  = 5; 
hidden_layer_size = 20;   
num_labels = 12; 

%fprintf('loading prev trained Neural Network Parameters ...\n')
load('weights.mat');

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
input = argv();
input = str2double(input);
pred = predict(Theta1,Theta2,input');
fprintf('\nClass Predicted: %f\n', pred);
#! /usr/bin/octave -qf

% depoyable nn by passing in command line args in the following manner:
% run_nn {flow} {bbox height} {center x} {center y} {depth}

input_layer_size  = 5;
hidden_layer_size = 20;
num_labels = 12;

%fprintf('loading prev trained Neural Network Parameters ...\n')
warning('off', 'Octave:load-file-in-path');
warning('off', 'all');
load('weights.mat');

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

input = argv();
input = str2double(input);
pred = predict(Theta1,Theta2,input');

switch (pred)
  case 1
    speed = 25;
  case 2
    speed = 30;
  case 3
    speed = 35;
  case 4
    speed = 40;
  case 5
    speed = 45;
  case 6
    speed = 50;
  case 7
    speed = 55;
  case 8
    speed = 60;
  case 9
    speed = 70;
  case 10
    speed = 80;
  case 11
    speed = 90;
  case 12
    speed = 95;
  otherwise
    speed = -1;
endswitch

fprintf('%d\n', speed);
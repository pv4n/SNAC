%% Initialization
clear ; close all; clc;


%% Setup the parameters
input_layer_size  = 7;  
hidden_layer1_size = 8;   
hidden_layer2_size = 6;
num_labels = 6;          

%% ================ Data Preperation ================
[X,y,m] = loadFilterAvg();


% Randomly shuffle
rand = randperm(size(X,1));
X = X(rand, :);
y=y(rand,:);
[X, mu, sigma] = featureNormalize(X);

% initialize theta1 and 2 (randinit already increments first param 
Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:)];
%pred = predict(T;(X(:,1),X(:,2))]


%% ================ Initializing Parameters ================
%fprintf('loading prev trained Neural Network Parameters ...\n')

initial_Theta1=Theta1;
initial_Theta2=Theta2;
initial_Theta3=Theta3;
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];



%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 400);

%  regularize
lambda = 0.01;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));
%
%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))): ...
                 (hidden_layer1_size * (input_layer_size + 1)) + ...
                (hidden_layer2_size * (hidden_layer1_size + 1))), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
                 
Theta3 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1)) + ...
                (hidden_layer2_size * (hidden_layer1_size + 1))):end), ...
                 num_labels, (hidden_layer2_size + 1));

                 
%% ================= Training Acc and Test =================
pred = predict(Theta1, Theta2, Theta3, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double((pred == y) * 100)));

weights = [Theta1(:) ; Theta2(:) ; Theta3(:)];  %unrolled
fname = strcat("weights_",strftime ("%m-%d_%H-%M", localtime (time ())));
fname = strcat(fname,".mat");
save(fname,"weights")


%% test with aXpY sets
data = load('/home/pavan/winlab/snac_ML/logs/static/car_a0p25.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*1;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n25 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

data = load('/home/pavan/winlab/snac_ML/logs/static/car_a0p30.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*2;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n30 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

data = load('/home/pavan/winlab/snac_ML/logs/static/car_a0p35.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*3;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n35 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

data = load('/home/pavan/winlab/snac_ML/logs/static/car_a0p40.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*4;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n40 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

data = load('/home/pavan/winlab/snac_ML/logs/static/car_a0p45.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*5;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n45 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

fprintf('\n now "testing" dynamic \n');

data = load('/home/pavan/winlab/snac_ML/logs/dynamic/car_a25p25.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*1;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n25_25 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

data = load('/home/pavan/winlab/snac_ML/logs/dynamic/car_a25p35.txt');
[X,y,m] = filter_outliers(data);
X=addNorm(X);
y=ones(size(X,1),1)*3;
X = (X .- mu) ./ sigma;
pred = predict(Theta1,Theta2, Theta3,X);
fprintf('\n25_35 Test Set Accuracy: %f\n', mean(double((pred == y) * 100)));

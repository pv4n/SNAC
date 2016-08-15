%% Initialization
clear ; close all; clc


%% Setup the parameters
input_layer_size  = 5;  % 2d0x20 Input Images of Digits
hidden_layer_size = 20;   % 25 hidden units
num_labels = 12;          

data=load('/home/pavan/winlab/snac_ML/logs/fbxyd_all.txt');
X=data(:,1:5);
%X=[X data(:,5)];
y=data(:,6);
m=size(X,1)
%get rid of outliers
_X = X(1,:);
_y=y(1,:);
for i=1:m
  if (X(i,3)>450 || X(i,1)<5 || X(i,3)<200)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];
    _y = [_y; y(i,:)];
  end
endfor
_X([1],:) = [];
_y([1],:) = [];

X=_X;
y=_y;
m=size(X,1)


[X mu sigma] = featureNormalize(X);
%X = [X mapFeature(X(:,2),X(:,1))];


% Randomly shuffle
rand_indices = randperm(m);
X = X(rand_indices, :);
%damn forgot to shuffle y!!!!!!!!!!!!!!!!!!!!!!!!
y=y(rand_indices,:);

%re-code y to 1: for the speeds
for i=1:m
  if (y(i)==25)
    y(i)=1;
  elseif (y(i)==30)
    y(i)=2;
  elseif (y(i)==35)
    y(i)=3;
  elseif (y(i)==40)
    y(i)=4;
  elseif (y(i)==45)
    y(i)=5;
  elseif (y(i)==50)
    y(i)=6;
  elseif y(i)==55
    y(i)=7;
  elseif y(i)==60
    y(i)=8;
  elseif y(i)==70
    y(i)=9;
  elseif y(i)==80
    y(i)=10;
  elseif y(i)==90
    y(i)=11;
  elseif y(i)==95
    y(i)=12;
  endif
end


% initialize theta1 and 2 (randinit already increments first param 
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];
%pred = predict(T;(X(:,1),X(:,2))]


%% ================ Initializing Pameters ================
%fprintf('loading prev trained Neural Network Parameters ...\n')
%load('Theta1.mat');
%load('Theta2.mat');

initial_Theta1=Theta1;
initial_Theta2=Theta2;
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 600);

%  regularize
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%% ================= Visualize Weights =================
%fprintf('\nVisualizing Neural Network... \n')
%displayData(Theta1(:, 2:end));


%% ================= Implement Predict =================
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(abs(pred - y))) * 100);

%% test with aXpY sets
data = load('/home/pavan/winlab/snac_ML/static/car_a55p0.txt');
X = data(:,1:5);
%X=[X data(:,5)];
m = size(X,1);

%get rid of outliers
_X = X(1,:);
for i=1:m
  if (X(i,3)>300 || X(i,1)<5 || X(i,3)<150)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];
  end
endfor

X=_X;
m=size(X,1);
y=ones(m,1)*7;

[X mu sigma] = featureNormalize(X);
%X = [X mapFeature(X(:,1),X(:,2))];
pred = predict(Theta1,Theta2,X);
fprintf('\n55 Test Set Accuracy: %f\n', mean(double(abs(pred - y))) );


data = load('/home/pavan/winlab/snac_ML/static/car_a45p0.txt');
X = data(:,2:6);
%X=[X data(:,5)];
m = size(X,1);

%get rid of outliers
_X = X(1,:);
for i=1:m
  if (X(i,3)>450 || X(i,1)<5 || X(i,3)<200)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];x   
  end
endfor

X=_X;
m=size(X,1);
y=ones(m,1)*5;

[X mu sigma] = featureNormalize(X);
%X = [X mapFeature(X(:,1),X(:,2))];
pred = predict(Theta1,Theta2,X);
fprintf('\n45 Test Set Accuracy: %f\n', mean(double(abs(pred - y))) );


data = load('/home/pavan/winlab/snac_ML/static/car_a35p0.txt');
X = data(:,1:5);
%X=[X data(:,5)];
m = size(X,1);

%get rid of outliers
_X = X(1,:);
for i=1:m
  if (X(i,3)>450 || X(i,1)<5 || X(i,3)<200)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];
  end
endfor

X=_X;
m=size(X,1);
y=ones(m,1)*3;

[X mu sigma] = featureNormalize(X);
%X = [X mapFeature(X(:,1),X(:,2))];
pred = predict(Theta1,Theta2,X);
fprintf('\n35 Test Set Accuracy: %f\n', mean(double(abs(pred - y))) );


data = load('/home/pavan/winlab/snac_ML/static/a25p0.txt');
X = data(:,1:5);
%X=[X data(:,5)];
m = size(X,1);

%get rid of outliers
_X = X(1,:);
for i=1:m
  if (X(i,3)>450 || X(i,1)<5 || X(i,3)<200)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];
  end
endfor

X=_X;
m=size(X,1);
y=ones(m,1)*1;

[X mu sigma] = featureNormalize(X);
%X = [X mapFeature(X(:,1),X(:,2))];
pred = predict(Theta1,Theta2,X);
fprintf('\n25 Test Set Accuracy: %f\n', mean(double(abs(pred - y))) );
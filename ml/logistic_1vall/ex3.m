%% Initialization
clear ; close all; clc

num_labels = 12;

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

data=load('/home/pavan/winlab/snacML/logs/fbxyd_all.txt');
X=data(:,1:5);
y = data(:, 6);
m = size(X, 1);

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
X = [X mapFeature(X(:,1),X(:,2))];

%re-code y to 1,2,3,4 for the speeds
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

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

%% ============ Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.9;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictOneVsAll(all_theta, X) == y)) * 100);

%% ================ Predict for One-Vs-All ================
%  for test dataset using trained thetas
data=load('/home/pavan/winlab/snacML/car_a55p0.txt');
X=data(:,1:5); 
[X mu sigma] = featureNormalize(X);
X = [X mapFeature(X(:,1),X(:,2))];
y=ones(size(X,1),1)*7;
pred = predictOneVsAll(all_theta, X);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y)) * 100);

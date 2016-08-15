%% this was the best try at regression. filtered, normalized, and polynomialed
%% best was r^2 .62

%% ================ Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

data=load('/home/pavan/winlab/snacML/fbxyd_2.txt');
X=data(:,1:3);
y = data(:, 6);
m = length(y);
%july17 get rid of outliers
%this init is the reason why theres one outlier still on graph
_X = X(1,:);
_y=y(1,:);
for i=1:m
  if (X(i,3)>500 || X(i,3)<200)  %outlier so taken out of matrix
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

% lets parse and store X according to speeds
X_25=zeros(1,3);X_35=X_25;X_45=X_35;X_55=X_45;X_60=X_55;X_70=X_60;X_80=X_70;
X_90=X_80;X_95=X_90;
y_25=[25];y_35=[35];y_45=[45];y_55=[55];y_60=[60];y_70=[70];y_80=[80];
y_90=[90];y_95=[95];
for i=1:m
  if (y(i)==25)
    X_25 = [X_25; X(i,:)];
    y_25 = [y_25; 25];
  elseif y(i)==35
    X_35 = [X_35; X(i,:)];
    y_35 = [y_35; 35];
  elseif y(i)==45
    X_45 = [X_45; X(i,:)];
    y_45 = [y_45; 45];
  elseif y(i)==55
    X_55 = [X_55; X(i,:)];
    y_55 = [y_55; 55];
  elseif y(i)==60
    X_60 = [X_60; X(i,:)];
    y_60 = [y_60; 60];
  elseif y(i)==70
    X_70 = [X_70; X(i,:)];
    y_70 = [y_70; 70];
  elseif y(i)==80
    X_80 = [X_80; X(i,:)];
    y_80 = [y_80; 80];
  elseif y(i)==90
    X_90 = [X_90; X(i,:)];
    y_90 = [y_90; 90];
  elseif y(i)==95
    X_95 = [X_95; X(i,:)];
    y_95 = [y_95; 95];
  end
endfor

[X mu sigma] = featureNormalize(X);
%plot flow, bbox
scatter3(X_25(2:end,1), X_25(2:end,2), X_25(2:end,3),"b", "s") %25
hold on;
xlabel("flow");ylabel("bbox");zlabel("center_x");
scatter3(X_35(2:end,1), X_35(2:end,2),X_35(2:end,3),"g", "s") %35 
hold on;
scatter3(X_45(2:end,1), X_45(2:end,2),X_45(2:end,3), "r","s") %45
hold on;
scatter3(X_55(2:end,1), X_55(2:end,2), X_55(2:end,3),"k","s") %55 
hold on;
scatter3(X_60(2:end,1), X_60(2:end,2), X_60(2:end,3),"b")
hold on;
scatter3(X_70(2:end,1), X_70(2:end,2), X_70(2:end,3),"g")
hold on;
scatter3(X_80(2:end,1), X_80(2:end,2), X_80(2:end,3),"r")
hold on;
scatter3(X_90(2:end,1), X_90(2:end,2), X_90(2:end,3),"k")
hold on;
scatter3(X_95(2:end,1), X_95(2:end,2), X_95(2:end,3),"b","x")

pause;
close all;

%% make X polynomial
Y = mapFeature(X(:,1),X(:,2));
X = [X Y];
%% Add intercept term to X
X = [ones(m, 1) X];
theta = normalEqn(X, y);
% training accuracy (r^2)
y_hat = zeros(m,1); %prediction for example i
for i=1:m
  in = X(i,:);
  y_hat(i) =  in * theta;
end

y_bar = sum(y_hat)/m;
r_squared = sum((y_hat.-y_bar).^2) / sum((y.-y_bar).^2)




%% Load "test" Data
data = load('/home/pavan/winlab/snacML/a25p0.txt');
Xtest = data(:, 1:3);
ytest = data(:,6);
[Xtest mu2 sigma2] = featureNormalize(Xtest);
b=length(data(:,5));
Xtest=[Xtest mapFeature(Xtest(:,1),Xtest(:,2))];
Xtest = [ones(b,1) Xtest];
prediction=zeros(b,1);
% test speed predictions on test set
for i=1:b
  in = Xtest(i,:);
  prediction(i) =  in * theta;
end
prediction;
%fprintf('\nTesting Set Accuracy (25): %f\n', mean(prediction .- ytest));

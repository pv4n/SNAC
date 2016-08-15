%% Initialization

%% ================ Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('d.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);
% make X polynomial
Y = mapFeature(X(:,1),X(:,2));
X = [X Y];






fprintf('resetting X to original data ...\n');
X=load('d.txt');
X=X(:,1:4);
%take derivatives
X=[X(1:m-2,1).-X(3:m,1) X(1:m-2,2).-X(3:m,2) X(1:m-2,3).-X(3:m,3) X(1:m-2,4).-X(3:m,4)];
%decrement m cuz we have derivative
m=m-2
%one less row for outputs y as well
y=y(3:m+2,:);

%july17 get rid of outliers
_X = X(1,:);
_y=y(1,:);
for i=1:m
%  if i>size(X,1)
%    break;
%  endif
  if (X(i,3)<-100 || X(i,3)>100 || X(i,1)>4)  %outlier so taken out of matrix
    %X([i],:) = [];
    continue;
  else
    _X = [_X; X(i,:)];
    _y = [_y; y(i,:)];
  end
endfor

X=_X;
y=_y;
m=size(X,1)

%%plot x,y,flow
%scatter3(X(1:302,3), X(1:302,4), X(1:302,1), "b") %25
%hold on;
%scatter3(X(303:490,3), X(303:490,4), X(303:490,1), "r") %35 
%hold on;
%scatter3(X(491:647,3), X(491:647,4), X(491:647,1), "g") %45
%hold on;
%scatter3(X(648:759,3), X(648:759,4), X(648:759,1), "k") %55 

%[X mu sigma] = featureNormalize(X);
%plot flow, bbox
scatter3(X(1:302,1), X(1:302,2), X(1:302,3),"b") %25
hold on;
xlabel("x");ylabel("y");zlabel("z");
scatter3(X(303:490,1), X(303:490,2),X(303:490,3),"r") %35 
hold on;
scatter3(X(491:647,1), X(491:647,2),X(491:647,3), "g") %45
hold on;
%scatter3(X(648:759,1), X(648:759,2), X(648:759,3),"k") %55 



%% make X polynomial
Y = mapFeature(X(:,1),X(:,2));
X = [X Y];
%% Add intercept term to X
X = [ones(m, 1) X];
theta = normalEqn(X, y);
% training accuracy (r^2)
y_hat = zeros(m,1); %prediction for example i
for i=1:m
  in = [X(i,1) X(i,2) X(i,3) X(i,4) X(i,5)];
  in = [in mapFeature(X(i,2),X(i,3))];
  y_hat(i) =  in * theta;
end

y_bar = sum(y_hat)/m;
r_squared = sum((y_hat.-y_bar).^2) / sum((y.-y_bar).^2)




%% Load "test" Data
data = load('/home/pavan/Desktop/logs/55.txt');
Xtest = data(:, 1:4);
[Xtest mu2 sigma2] = featureNormalize(Xtest);
b=length(data(:,5));
Xtest = [ones(b,1) Xtest];
size(Xtest,2);
prediction=zeros(b,1);
% test speed predictions on test set
for i=1:b
  in = [Xtest(i,1) Xtest(i,2) Xtest(i,3) Xtest(i,4) Xtest(i,5)];
  in = [in mapFeature(Xtest(i,2),Xtest(i,3))];
  prediction(i) =  in * theta;
end



%in = [1 ((X(i,1) - mu(:,1))/sigma(:,1)) ((96 - mu(:,2))/sigma(:,2)) ((58 - mu(:,3))/sigma(:,3)) ((364 - mu(:,4))/sigma(:,4))];
%  in = [in mapFeature(((12 - mu(:,1))/sigma(:,1)),((96 - mu(:,2))/sigma(:,2)))];

% ============================================================

%fprintf(['Predicted speed for a given normalized flow, bbox, centerx, centery, mapped columns for flow and bbox ' ...
%         '(using normaleqn):\n %f\n'], speed);


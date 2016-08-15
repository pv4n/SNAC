clear; close all;

data=load('/home/pavan/winlab/snac_ML/logs/db_728.txt');
X=data(:,2:4);
y = data(:, 7);
m = length(y);

%get rid of outliers
%this init is the reason why theres one outlier still on graph
_X = X(1,:);
_y=y(1,:);
for i=1:m
  if (X(i,3)>400 || X(i,3)<200 || X(i,1)<3)  %outlier so taken out of matrix
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

%[X mu sigma] = featureNormalize(X);

% lets parse and store X according to speeds
X_25=zeros(1,3);X_35=X_25;X_45=X_35;X_55=X_45;X_60=X_55;X_70=X_60;X_80=X_70;
X_90=X_80;X_95=X_90;X_30=X_25;X_40=X_25;X_50=X_25;
for i=1:m
  if (y(i)==25)
    X_25 = [X_25; X(i,:)];
  elseif y(i)==30
    X_30 = [X_30; X(i,:)];
  elseif y(i)==35
    X_35 = [X_35; X(i,:)];
  elseif y(i)==40
    X_40 = [X_40; X(i,:)];
  elseif y(i)==45
    X_45 = [X_45; X(i,:)];
  elseif y(i)==50
    X_50 = [X_50; X(i,:)];
  elseif y(i)==55
    X_55 = [X_55; X(i,:)];
  elseif y(i)==60
    X_60 = [X_60; X(i,:)];
  elseif y(i)==70
    X_70 = [X_70; X(i,:)];
  elseif y(i)==80
    X_80 = [X_80; X(i,:)];
  elseif y(i)==90
    X_90 = [X_90; X(i,:)];
  elseif y(i)==95
    X_95 = [X_95; X(i,:)];
  end
endfor

%% moving avg on each speed
X_25 = mave(X_25,1);

%% calculating NORMS
%insert zeros in beginning as we are going sequential
X_25 = [zeros(size(X_25,2)) ; X_25];
%add col to store new data
X_25 = [X_25 zeros(size(X_25,1))];
for i=2:size(X_25,1)
%  X_25(i,4) = ((X_25(i,1) - X_25(i-1,1))^2 + ((X_25(i,3) - X_25(i-1,3))^2))^.5;
  X_25(i,4) = norm(X_25(i,1)- X_25(i-1,1)) + norm(X_25(i,1)- X_25(i-1,1));
endfor
%get rid of zeros row and second cuz it would use the first
X_25([1],:)=[];
X_25([1],:)=[];

X_35 = [zeros(size(X_35,2)) ; X_35];
%add col to store new data
X_35 = [X_35 zeros(size(X_35,1))];
for i=2:size(X_35,1)
%  X_35(i,4) = ((X_35(i,1) - X_35(i-1,1))^2 + ((X_35(i,3) - X_35(i-1,3))^2))^.5;
  X_35(i,4) = norm(X_35(i,1)- X_35(i-1,1)) + norm(X_35(i,1)- X_35(i-1,1));
endfor
X_35([1],:)=[];
X_35([1],:)=[];

X_45 = [zeros(size(X_45,2)) ; X_45];
%add col to store new data
X_45 = [X_45 zeros(size(X_45,1))];
for i=2:size(X_45,1)
%  X_45(i,4) = ((X_45(i,1) - X_45(i-1,1))^2 + ((X_45(i,3) - X_45(i-1,3))^2))^.5;
  X_45(i,4) = norm(X_45(i,1)- X_45(i-1,1)) + norm(X_45(i,1)- X_45(i-1,1));
endfor
X_45([1],:)=[];
X_45([1],:)=[];

X_55 = [zeros(size(X_55,2)) ; X_55];
%add col to store new data
X_55 = [X_55 zeros(size(X_55,1))];
for i=2:size(X_55,1)
%  X_55(i,4) = ((X_55(i,1) - X_55(i-1,1))^2 + ((X_55(i,3) - X_55(i-1,3))^2))^.5;
  X_55(i,4) = norm(X_55(i,1)- X_55(i-1,1)) + norm(X_55(i,1)- X_55(i-1,1));
endfor
X_55([1],:)=[];
X_55([1],:)=[];

X_60 = [zeros(size(X_60,2)) ; X_60];
%add col to store new data
X_60 = [X_60 zeros(size(X_60,1))];
for i=2:size(X_60,1)
%  X_95(i,4) = ((X_95(i,1) - X_95(i-1,1))^2 + ((X_95(i,3) - X_95(i-1,3))^2))^.5;
  X_60(i,4) = norm(X_60(i,1)- X_60(i-1,1)) + norm(X_60(i,1)- X_60(i-1,1));
endfor
X_60([1],:)=[];
X_60([1],:)=[];

%X_95 = [zeros(size(X_95,2)) ; X_95];
%%add col to store new data
%X_95 = [X_95 zeros(size(X_95,1))];
%for i=2:size(X_60,1)
%%  X_95(i,4) = ((X_95(i,1) - X_95(i-1,1))^2 + ((X_95(i,3) - X_95(i-1,3))^2))^.5;
%  X_95(i,4) = norm(X_95(i,1)- X_95(i-1,1)) + norm(X_95(i,1)- X_95(i-1,1));
%endfor
%X_95([1],:)=[];
%X_95([1],:)=[];


%% DRAWING NORMS
scatter(X_25(:,4), X_25(:,3), "b");
hold on;
scatter(X_35(:,4), X_35(:,3), "g");
hold on;
scatter(X_45(:,4), X_45(:,3), "r");
hold on;
scatter(X_55(:,4), X_55(:,3), "k");

%% DRAWING RAW DATA
%%[X mu sigma] = featureNormalize(X);
%%plot flow, bbox, centerx
%scatter3(X_25(2:end,1), X_25(2:end,2), X_25(2:end,3),"b", "s") %25
%hold on;
%scatter3(X_30(2:end,1), X_30(2:end,2),X_30(2:end,3),"g", "s") %35 
%hold on;
%scatter3(X_35(2:end,1), X_35(2:end,2),X_35(2:end,3),"r", "s") %35 
%hold on;
%scatter3(X_40(2:end,1), X_40(2:end,2),X_40(2:end,3), "k","s") %45
%hold on;
%scatter3(X_45(2:end,1), X_45(2:end,2),X_45(2:end,3), "b") %45
%hold on;
%scatter3(X_50(2:end,1), X_50(2:end,2), X_50(2:end,3),"g") %55 
%hold on;
%scatter3(X_55(2:end,1), X_55(2:end,2), X_55(2:end,3),"r") %55 
%hold on;
%scatter3(X_60(2:end,1), X_60(2:end,2), X_60(2:end,3),"k")
%hold on;
%scatter3(X_70(2:end,1), X_70(2:end,2), X_70(2:end,3),"b","x")
%hold on;
%scatter3(X_80(2:end,1), X_80(2:end,2), X_80(2:end,3),"g","x")
%hold on;
%scatter3(X_90(2:end,1), X_90(2:end,2), X_90(2:end,3),"r","x")
%hold on;
%scatter3(X_95(2:end,1), X_95(2:end,2), X_95(2:end,3),"k","x")
%xlabel("flow");ylabel("bbox");zlabel("center_x");
%h = legend ({'25'}, '30','35','40','45','50','55','60','70','80','90','95');



%%% trying to plot old data on top of new
%
%pause;
%close all;
%
%
%% plots to make sure 60's are the same for 25+35, 35+25, 30+30
%
%data=load('/home/pavan/winlab/snacML/logs/car_a25p35.txt');
%_data = data(1,:);
%for i=1:size(data,1)
%  if (data(i,3)>500 || data(i,3)<200)  %outlier so taken out of matrix
%    continue;
%  else
%    _data = [_data; data(i,:)];
%  end
%endfor
%_data([1],:) = [];
%data=_data;
%scatter(data(:,1),data(:,3),"b")
%hold on;
%
%data=load('/home/pavan/winlab/snacML/logs/car_a35p25.txt');
%_data = data(1,:);
%for i=1:size(data,1)
%  if (data(i,3)>500 || data(i,3)<200)  %outlier so taken out of matrix
%    continue;
%  else
%    _data = [_data; data(i,:)];
%  end
%endfor
%_data([1],:) = [];
%data=_data;
%scatter(data(:,1),data(:,3),"r","s")

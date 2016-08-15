function [X] = mave(X,n)

%lets do moving average over n rows of matrix X 
m=size(X,1);


for i=1:n
  % put zeros to make averaging happy
  X = [zeros(1, size(X,2)); X];
endfor

for i=(n):m+n
  X(i,:) = (sum(X(i-(n-1):i,:))) ./ n;    
endfor

for i=1:n
  % taking out the rows first put in
  X([1],:) = [];
endfor


endfunction
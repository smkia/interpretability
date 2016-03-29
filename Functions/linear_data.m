function [X, Y] = linear_data(N, m, b, noise, S)
% Sample a dataset from a linear separable dataset

%    INPUT 
%       N      1x2 vector that fix the numberof samples from each class
%       m      slope of the separating line. Default is random.    
%       b      bias of the line. Default is random.
%       noise  true or false that specifies noise addition.
%       S      Covariance of Noise.
%    OUTPUT
%       X      data matrix with a sample for each row 
%       Y      vector with the labels
%
%   EXAMPLE:
%       [X, Y] = linearData([10, 10]);


if (nargin < 5)
	S = [0.02,-0.01;-0.01,1];
end
if (nargin < 4)
	noise = 0;
end
if (nargin < 3)
	b = rand()*0.5;
end
if (nargin < 2)
	m = rand() * 2 +0.01;
end

X = [];
while(size(X,1) < N(1))
    xx = rand(1);
    yy = rand(1);
    fy = xx*m + b;
    if(yy < fy - 0.015)
        if noise
            X = [X;[xx, yy]+ [chol(S)'*randn(2,1)]'];
        else
            X = [X;[xx, yy]];
        end
    end
end

while(size(X,1) < sum(N))
    xx = rand(1);
    yy = rand(1);
    fy = xx*m + b;
    if(yy > fy + 0.015)
        if noise
            X = [X;[xx, yy]+ [chol(S)'*randn(2,1)]'];
        else
            X = [X;[xx, yy]];
        end
    end
end

Y = ones(sum(N), 1);
Y(1:N(1))  = -1;

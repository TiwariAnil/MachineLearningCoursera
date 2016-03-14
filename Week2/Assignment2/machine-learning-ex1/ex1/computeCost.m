function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Formula:  x(i) * theta - y(i);
%cost =0 ;

%for i=1:m
%  cost = cost + ( (theta(1,1)* X(i,1) + theta(2,1)*X(i,2)) - y(i,1) )^2 ;
%end;

%cost = cost / (2*m);

hypothisis = X * theta;

cost_matrix = ( hypothisis - y );

square_deviation = cost_matrix.^2; % each element ^2



J = 1/ (2*m) * sum(square_deviation);
% =========================================================================

end

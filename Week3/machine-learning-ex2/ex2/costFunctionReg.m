function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;  % the cost fucntion
grad = zeros(size(theta));  % the gard i.e d/dx (J)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




% calculate cost function i.e h(x) = g(z) = 1./(1+e^-z)

hypothisis = X * theta;
g = sigmoid(hypothisis);
cost_mat = (( y .* log(g) ) + ( 1 - y) .* log(1 - g));
cost = sum(cost_mat);

% calculate penalty
% we need not penaliize the theta(1) as its just the base theta we have given.
% excluded the first theta value
penalty = sum(theta(2:end).^2);
penalty = penalty*lambda/(2*m);
J = cost * (-1) / m;
% Regularization to avoid OVERFITTING/UNDERFITTING 
%Add penalty to the cost!
J = J + penalty;

grad = (X' * ((g) - y))/m; 
%Adding the base value ie theta(1) = 0
theta = [0 ; theta(2:end, :)];
theta = theta .* (lambda/m);

grad = grad + theta;

% =============================================================

end

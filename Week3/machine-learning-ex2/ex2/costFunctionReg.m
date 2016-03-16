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

% ====================== CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothisis = X*theta;   % Linear Reg function H()

G = sigmoid(hypothisis);  % Logistic Reg funciton g() 

cost = 0;
theta_change = 0;
for i=1:m
  cost = cost + (  (y(i) * log(G(i)) )   +    ( 1 - y(i)) * log(1 - G(i)) );
  %Stheta_change = theta_change + theta(i).^2;
end
theta_change = theta.^2;
theta_change(1) = theta(1);
theta_change = sum(theta_change) - theta(1); %
% above is coz we don't have to regularize the theta(1) i.e theta0
% this we add extra not coz any feature 
theta_change = (lambda* theta_change)/ (2) / m ;
J = cost * (-1) / m;
J = J + theta_change;

grad = (X' * ((G) - y))/m; 

for i=2:size(theta)
  grad(i) = grad(i) + (lambda / m) * theta(i);
end;

% =============================================================
end
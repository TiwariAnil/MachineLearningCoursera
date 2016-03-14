function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
t1 = theta(1,1);
t2 = theta(2,1);
val1 = 0;
val2 = 0;

common = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
 
%%%%%%%%%% OLD
%    t1 = theta(1,1);
%    t2 = theta(2,1);
%    temp1 = 0;
%    temp2 = 0;
%    for i = 1:m
    
%        common = (t1* X(i,1) + t2*X(i,2)) - y(i,1) ;        
%        temp1 = temp1 + common;
%        temp2 = temp2 + ( common * X(i,2) );         
%    end;

%    theta(1,1) = t1 - ( alpha * temp1 / m );
 %   theta(2,1) = t2 - ( alpha * temp2 / m ) ;
    
    hypothisis = X * theta;
    theta0 = theta(1) - alpha / m * sum( ( hypothisis - y).* X(:,1) );  
    theta1 = theta(2) - alpha / m * sum( ( hypothisis - y).* X(:,2) );  
    
    theta(1) = theta0;
    theta(2) = theta1;   
    
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); %5000
n = size(X, 2); %400

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
%===================Anil Code===============================

% Since we have 10 digits i.e class for which we need theta, (A diff theta for all)
% So we need a matrix of theta for each digit.

% For each digit, we need to train theta i.e 1, 2, 3....9, 10(as 0) , 
for i= 1:num_labels

    % set the theta for 'i' digit as zeros number_of_columns, as number of rows;
    % each image has 20x20 i.e 400 pixels, so we need theta of size 400 i.e one "weight/theta" for each "input" i.e pixel.
    initial_theta = zeros(n+1,1);
    %theta is ROW Vector [ 0;0;0;0...n+1]
    
    %     % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);
     
    % HOLD-ON: We are given 5000 images of 20x20 pixels each. We need theta that will form H(x) using theta to give the
    % final output as DIGIT match 
     
    %     % Run fmincg to obtain the optimal theta
    %     % This function will return theta and the cost 

    %So we need, theta after learning on X = input and  y = ouput(only the digit 'i') , we pass options, initial_theta
    % Here, while passing parameters in lrCostFunction
    % y==i means, it will set all values in y as 1 if it is equal to 'i' else it will set 0.
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), initial_theta, options);

    %Just store this theta at ith position or ith ROW in the all_theta matrix
    % Since, theta is ruturned in form of ROW VECTOR, we need to store in COLUMN VECTOR FORM, :D Transpose!
    all_theta(i,:) = theta';

end

% ==================TIP-=======================================================
%{
Octave/MATLAB Tip: Logical arrays in Octave/MATLAB are arrays
which contain binary (0 or 1) elements. In Octave/MATLAB, evaluating
the expression a == b for a vector a (of size m1) and scalar b will return
a vector of the same size as a with ones at positions where the elements
of a are equal to b and zeroes where they are dierent. To see how this
works for yourself, try the following code in Octave/MATLAB:
a = 1:10; % Create a and b
b = 3;
a == b % You should try different values of b here
%}

end

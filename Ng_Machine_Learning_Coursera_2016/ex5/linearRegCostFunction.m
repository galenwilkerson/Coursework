function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% COST J:
h = X * theta;
h_minus_y = h - y;

J = h_minus_y' * h_minus_y;
J = J/(2*m);

% REGULARIZED COST (don't regularize the first element)
theta(1) = 0;
reg_term = theta' * theta;
reg_term = reg_term * (lambda/(2*m));
J = J + reg_term;

% GRADIENT 
% [1 x 2] = [1 x 12] * [12 x 2]
grad = (1/m) * h_minus_y' * X;

% REGULARIZED GRADIENT  (don't regularize the first element)
% [1 x 2] = [1 x 2] + [1 x 1] * [1 x 2]
grad = grad + (lambda/m) * theta';

% =========================================================================

grad = grad(:);

end

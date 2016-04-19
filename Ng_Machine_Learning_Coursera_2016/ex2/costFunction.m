function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = size(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = 0;

% the sigmoid values (predictions) for each data point
% 100 x 1 = sigmoid(100 x 3 * 3 X 1) = sigmoid(100 x 1)

h = sigmoid(X * theta);
log_h = log(h);
log_1_minus_h = log(1 - h);
one_minus_y = 1 - y;

% vector versions of J equation and gradient dJ_d_theta:
% J = (1/m) * - y' * log_h - one_minus_y' * log_1_minus_h

J = (-1/m) * ((log_h)'*y + log_1_minus_h'*(1-y))


h_minus_y = h - y;
grad = (1/m) * X' * h_minus_y;

% =============================================================

end
